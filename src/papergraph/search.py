"""Paper search over a corpus: the product pipeline, shared by the web API and the agent.

    hybrid (BM25 + bge-small, RRF) -> cross-encoder rerank of the top 20 -> paper records

These defaults are the M4/M5 winners on LitSearch dev (see reports/m4, reports/m5).
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import psycopg

from papergraph.encoders import get_model
from papergraph.rerank import cross_encoder, merge_reranked
from papergraph.retrieval import dense, fusion
from papergraph.retrieval.filters import Filters

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


def search(
    conn: psycopg.Connection,
    query: str,
    k: int = 10,
    filters: Filters | None = None,
    rerank: bool = True,
) -> list[Paper]:
    """The top-k papers for a free-text interest, best first."""
    if not query.strip():
        return []
    ids = fusion.search(
        conn, query, max(k, 100), retrievers=HYBRID, candidates=100, filters=filters
    )
    if rerank and ids:
        top = ids[:RERANK_TOP_N]
        ids = merge_reranked(ids, cross_encoder.rerank(conn, query, top))
    return fetch_papers(conn, ids[:k])


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
