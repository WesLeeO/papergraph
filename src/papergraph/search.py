"""Paper search over a corpus: the product pipeline, shared by the web API and the agent.

    hybrid (BM25 + bge-small, RRF) -> cross-encoder rerank of the top 20 -> paper records

These defaults are the M4/M5 winners on LitSearch dev (see reports/m4, reports/m5).
"""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

import psycopg
from psycopg import sql

from papergraph.encoders import get_encoder, get_model
from papergraph.query import Understood, understand
from papergraph.rerank import cross_encoder, merge_reranked
from papergraph.retrieval import dense, fusion
from papergraph.retrieval.filters import Filters, filter_sql

HYBRID: list[dict[str, Any]] = [
    # max_df=0.1 skips query terms in >10% of papers: ~6-10x faster BM25 on the
    # 581k corpus, and no measured loss for this full pipeline on LitSearch dev
    # (nDCG@10 -0.000 [-0.018, +0.017]; run 20261103T170554Z_product-maxdf01).
    {"kind": "bm25", "params": {"k1": 1.2, "b": 0.75, "max_df": 0.1}},
    {
        "kind": "dense",
        "params": {"model": "bge-small", "query_instruction": True, "ef_search": 100},
    },
]
RERANK_TOP_N = 20
# Candidates per retriever when ranking a query for browsing. Up to twice this
# many papers come out of fusion: the "end of results" a user can scroll to.
POOL = 300
# A run of words counts as an author name only if it matches someone with at
# least this many papers: fewer, and topic words could pass for rare names.
MIN_AUTHOR_PAPERS = 2

T = TypeVar("T")


class _TTLCache:
    """A small thread-safe LRU cache with expiry, for ranked id lists.

    Scrolling asks for page after page of the same query; ranking it once and
    slicing the cached list keeps pages fast and consistent with each other.
    Entries expire so a re-ingested corpus shows up within minutes.
    """

    def __init__(self, max_entries: int = 256, ttl_s: float = 600) -> None:
        self.max_entries, self.ttl_s = max_entries, ttl_s
        self._data: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_compute(self, key: Hashable, compute: Callable[[], T]) -> T:
        now = time.monotonic()
        with self._lock:
            hit = self._data.get(key)
            if hit is not None and now - hit[0] < self.ttl_s:
                self._data.move_to_end(key)
                return hit[1]  # type: ignore[no-any-return]
        value = compute()  # outside the lock: ranking takes ~0.5 s
        with self._lock:
            self._data[key] = (now, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_rankings = _TTLCache()


def _corpus_key(conn: psycopg.Connection) -> str:
    """Which corpus the connection searches (public, bench_litsearch, ...)."""
    row = conn.execute("SHOW search_path").fetchone()
    return str(row[0]) if row else ""


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    abstract: str
    authors: list[str]
    published: str | None  # ISO date of arXiv v1
    venue: str | None
    venue_year: int | None
    categories: list[str]

    @property
    def url(self) -> str:
        # LitSearch ids are Semantic Scholar corpus ids (all digits); arXiv ids have '.' or '/'.
        if self.id.isdigit():
            return f"https://www.semanticscholar.org/p/{self.id}"
        return f"https://arxiv.org/abs/{self.id}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "url": self.url}


_PAPERS = """
SELECT p.id, p.title, p.abstract, p.published, p.venue, p.venue_year,
       coalesce((SELECT array_agg(a.name ORDER BY a.position)
                 FROM paper_authors a WHERE a.paper_id = p.id), '{}'),
       coalesce((SELECT array_agg(c.category ORDER BY c.category)
                 FROM paper_categories c WHERE c.paper_id = p.id), '{}')
FROM papers p
WHERE p.id = ANY(%s)
"""


def fetch_papers(conn: psycopg.Connection, ids: Sequence[str]) -> list[Paper]:
    """Paper records for `ids`, in the same order (unknown ids are skipped)."""
    if not ids:
        return []
    rows = {
        r[0]: Paper(
            id=r[0],
            title=r[1],
            abstract=r[2],
            published=r[3].isoformat() if r[3] else None,
            venue=r[4],
            venue_year=r[5],
            authors=list(r[6]),
            categories=list(r[7]),
        )
        for r in conn.execute(_PAPERS, (list(ids),))
    }
    return [rows[i] for i in ids if i in rows]


def ranked_ids(
    conn: psycopg.Connection,
    query: str,
    filters: Filters | None = None,
    rerank: bool = True,
) -> list[str]:
    """Every paper id the pipeline returns for `query`, best first (cached).

    Hybrid retrieval over a deep pool (POOL candidates per retriever), with
    the top RERANK_TOP_N reordered by the cross-encoder.
    """
    if not query.strip():
        return []
    f = filters or Filters()

    def compute() -> list[str]:
        ids = fusion.search(conn, query, 2 * POOL, retrievers=HYBRID, candidates=POOL, filters=f)
        if rerank and ids:
            ids = merge_reranked(ids, cross_encoder.rerank(conn, query, ids[:RERANK_TOP_N]))
        return ids

    key = ("search", _corpus_key(conn), query.strip(), f, rerank)
    return _rankings.get_or_compute(key, compute)


def search(
    conn: psycopg.Connection,
    query: str,
    k: int = 10,
    filters: Filters | None = None,
    rerank: bool = True,
    offset: int = 0,
) -> list[Paper]:
    """Papers ranked offset..offset+k for a free-text interest, best first."""
    ids = ranked_ids(conn, query, filters, rerank)
    return fetch_papers(conn, ids[offset : offset + k])


def get_paper(conn: psycopg.Connection, paper_id: str) -> Paper | None:
    papers = fetch_papers(conn, [paper_id])
    return papers[0] if papers else None


def similar_papers(
    conn: psycopg.Connection, paper_id: str, k: int = 10, filters: Filters | None = None
) -> list[Paper]:
    """Papers whose embeddings are nearest to this paper's ("more like this")."""
    m = get_model("bge-small")
    row = conn.execute(
        f"SELECT embedding::text FROM {m.table} WHERE paper_id = %s", (paper_id,)
    ).fetchone()
    if row is None:
        return []
    ids = dense.nearest(conn, row[0], k, m, filters=filters, exclude=paper_id)
    return fetch_papers(conn, ids)


def venues(conn: psycopg.Connection) -> list[tuple[str, int]]:
    """Detected venues with paper counts, most common first (for filter menus)."""
    rows = conn.execute(
        "SELECT venue, count(*) FROM papers WHERE venue IS NOT NULL GROUP BY venue ORDER BY 2 DESC"
    )
    return [(v, int(n)) for v, n in rows]


# --- authors ------------------------------------------------------------------


def _tokens(name: str) -> list[str]:
    """'Y. LeCun' -> ['y', 'lecun']: lowercase words, dots and commas dropped."""
    return [t for t in name.lower().replace(".", " ").replace(",", " ").split() if t]


def name_matches(query: str, candidate: str) -> bool:
    """Whether an author name in the data plausibly refers to the queried person.

    Same surname, and if the query has a first name, the candidate's first
    name agrees with it or is its initial:
    'Yann LeCun' matches 'Yann LeCun', 'Yann Lecun', 'Y. LeCun' and
    'Yann André LeCun', but not 'Le Cam' or 'Xavier LeCun'. Surnames are
    compared without spaces, so 'Yann Le Cun' matches 'Yann LeCun' and
    'van der Maaten' matches 'Laurens van der Maaten'. Names written surname
    first ('Li Fei-Fei' for 'Fei-Fei Li') match too.
    """
    q, c = _tokens(query), _tokens(candidate)
    if not q or not c:
        return False
    return _same_person(q, c) or (len(q) == 2 and _same_person(q[::-1], c))


def _surnames(tokens: list[str]) -> set[str]:
    """Possible surnames: every run of trailing words, spaces removed.
    ['yann', 'andré', 'lecun'] -> {'andrélecun', 'lecun'}."""
    if len(tokens) == 1:
        return {tokens[0]}
    return {"".join(tokens[j:]) for j in range(1, len(tokens))}


def _same_person(q: list[str], c: list[str]) -> bool:
    c_surnames = _surnames(c)
    if "".join(q) in c_surnames:  # surname only ('LeCun', 'Le Cun'): any first name
        return True
    if len(q) == 1 or not (_surnames(q) & c_surnames):
        return False
    first_q, first_c = q[0], c[0]
    if len(first_q) == 1 or len(first_c) == 1:  # an initial on either side
        return first_q[0] == first_c[0]
    return first_q == first_c


def find_authors(conn: psycopg.Connection, name: str, limit: int = 10) -> list[tuple[str, int]]:
    """Name variants in the data matching `name`, with paper counts, most papers first."""
    tokens = _tokens(name)
    longest = max(tokens, key=len) if tokens else ""
    # Need two letters/digits: 'a' would match millions of names, and '%%%%'
    # has no trigrams for the index to use, so it would scan every row.
    if sum(ch.isalnum() for ch in longest) < 2:
        return []
    # Escape LIKE wildcards typed by the user: '%' must not match every name.
    pattern = longest.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    # Narrow candidates with the longest name part: the most selective, and it
    # must appear in any spelling or order ('fei-fei' for 'Fei-Fei Li'). The
    # trigram index on lower(name) makes this LIKE fast.
    rows = conn.execute(
        "SELECT name, count(*) FROM paper_authors WHERE lower(name) LIKE %s "
        "GROUP BY name ORDER BY count(*) DESC LIMIT 200",
        (f"%{pattern}%",),
    )
    return [(n, int(c)) for n, c in rows if name_matches(name, n)][:limit]


def author_ranked_ids(
    conn: psycopg.Connection,
    name: str,
    query: str | None = None,
    filters: Filters | None = None,
) -> tuple[list[tuple[str, int]], list[str]]:
    """Name variants matched, and the ids of all their papers (cached).

    Without `query`: newest first. With `query`: ranked by relevance to it
    (embedding similarity, then the cross-encoder on the top 20). An author
    has at most a few hundred papers, so they are ranked exactly instead of
    searching the whole corpus with a very selective filter.
    """
    f = filters or Filters()
    q = (query or "").strip()

    def compute() -> tuple[list[tuple[str, int]], list[str]]:
        variants = find_authors(conn, name)
        if not variants:
            return [], []
        condition, params = filter_sql(f, "p")
        rows = conn.execute(
            sql.SQL(
                "SELECT p.id FROM papers p WHERE {} AND EXISTS (SELECT 1 FROM paper_authors a "
                "WHERE a.paper_id = p.id AND a.name = ANY(%(names)s)) "
                "ORDER BY p.published DESC NULLS LAST, p.id LIMIT 2000"
            ).format(condition),
            {**params, "names": [n for n, _ in variants]},
        )
        ids = [r[0] for r in rows]
        if q and ids:
            m = get_model("bge-small")
            embedded = get_encoder(m.name).encode_query(dense.query_text(q, m, True))
            vec = dense.vector_literal(embedded)
            ranked = [
                r[0]
                for r in conn.execute(
                    sql.SQL(
                        "SELECT paper_id FROM {} WHERE paper_id = ANY(%s) "
                        "ORDER BY embedding <=> %s::vector"
                    ).format(sql.Identifier(m.table)),
                    (ids, vec),
                )
            ]
            # Papers without an embedding yet keep their date order, after the ranked ones.
            seen = set(ranked)
            ids = ranked + [i for i in ids if i not in seen]
            ids = merge_reranked(ids, cross_encoder.rerank(conn, q, ids[:RERANK_TOP_N]))
        return variants, ids

    key = ("author", _corpus_key(conn), name.strip().lower(), q, f)
    return _rankings.get_or_compute(key, compute)


def author_papers(
    conn: psycopg.Connection,
    name: str,
    query: str | None = None,
    k: int = 15,
    filters: Filters | None = None,
    offset: int = 0,
) -> tuple[list[tuple[str, int]], list[Paper]]:
    """Papers by an author (all spellings of the name), with the variants found."""
    variants, ids = author_ranked_ids(conn, name, query, filters)
    return variants, fetch_papers(conn, ids[offset : offset + k])


# --- understood queries ----------------------------------------------------------


def browse_ids(conn: psycopg.Connection, filters: Filters) -> list[str]:
    """Papers matching filters only (no topic, no author), newest first (cached)."""

    def compute() -> list[str]:
        condition, params = filter_sql(filters, "p")
        rows = conn.execute(
            sql.SQL(
                "SELECT p.id FROM papers p WHERE {} ORDER BY p.published DESC NULLS LAST, p.id "
                "LIMIT 2000"
            ).format(condition),
            params,
        )
        return [r[0] for r in rows]

    return _rankings.get_or_compute(("browse", _corpus_key(conn), filters), compute)


def _venue_names(conn: psycopg.Connection) -> list[str]:
    return _rankings.get_or_compute(
        ("venues", _corpus_key(conn)), lambda: [v for v, _ in venues(conn)]
    )


@dataclass(frozen=True)
class QueryResult:
    understood: Understood
    author_name: str | None  # the author searched (parsed or explicit), as typed
    matched_authors: list[tuple[str, int]]  # spellings found for it
    filters: Filters  # parsed and explicit filters combined
    ids: list[str]  # the full ranking; pages are slices of it


def run_query(
    conn: psycopg.Connection,
    text: str,
    *,
    author: str = "",
    filters: Filters | None = None,
    skip: Sequence[str] = (),
    rerank: bool = True,
) -> QueryResult:
    """Understand `text` (papergraph.query), then rank papers accordingly.

    `author` and `filters` come from the filter panel and take precedence over
    what the parser found in the text.
    """
    explicit = filters or Filters()
    skipped = set(skip) | ({"author"} if author.strip() else set())
    memo: dict[str, bool] = {}

    def is_author(name: str) -> bool:
        if name not in memo:  # one lookup per candidate name
            variants = find_authors(conn, name)
            memo[name] = bool(variants) and variants[0][1] >= MIN_AUTHOR_PAPERS
        return memo[name]

    u = understand(text, venues=_venue_names(conn), is_author=is_author, skip=skipped)
    f = Filters(
        year_from=explicit.year_from or u.year_from,
        year_to=explicit.year_to or u.year_to,
        venues=explicit.venues or u.venues,
        categories=explicit.categories,
        venue_year=u.venue_year if not explicit.venues else None,
    )
    name = author.strip() or u.author or ""
    topic = u.topic
    matched: list[tuple[str, int]] = []
    if name:
        matched, ids = author_ranked_ids(conn, name, topic or None, f)
    elif topic:
        ids = ranked_ids(conn, topic, f, rerank)
    elif not f.empty:
        ids = browse_ids(conn, f)
    else:
        ids = []
    if u.arxiv_id and get_paper(conn, u.arxiv_id) is not None:
        ids = [u.arxiv_id, *(i for i in ids if i != u.arxiv_id)]
    return QueryResult(u, name or None, matched, f, ids)
