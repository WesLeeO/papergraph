"""LitSearch: fetch, split and load into schema bench_litsearch. Run with `make bench-litsearch`.

LitSearch (Ajith et al., 2024) has 597 literature-search questions written about
ML/NLP papers, each labelled with the 1-5 papers that answer it, over a corpus
of 64,183 papers identified by Semantic Scholar corpus ids.

- Fetch: the Hugging Face parquet files hold full paper text (~2.9 GB). Parquet
  is stored column by column, so reading only the columns we need over HTTP
  range requests transfers a small fraction of that. The result is cached
  under data/ and the dataset revision is pinned, so every run sees the same data.
- Split: LitSearch ships one "full" split, and tuning on test is not allowed.
  We make a dev/test split ourselves; see `assign_splits`. It is committed in
  litsearch_split.json and checked on every load.
- Load: the corpus goes into bench_litsearch.papers through the same schema as
  the arXiv corpus, so the same retrieval code runs on both. Embed it afterwards
  with `python -m papergraph.ingest.embed --schema bench_litsearch`.
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from papergraph.db import connect
from papergraph.ingest.arxiv import clean_text
from papergraph.migrate import migrate

REPO = "princeton-nlp/LitSearch"
# Pinned dataset commit: logged with every run as the corpus snapshot id.
REVISION = "9573fb284a1026c998df47024b888a163f0f0e25"
SCHEMA = "bench_litsearch"
CACHE = Path("data/benchmarks/litsearch")
SPLIT_FILE = Path(__file__).with_name("litsearch_split.json")
DEV_FRACTION = 1 / 3

SPECIFICITY = {0: "broad", 1: "specific"}


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    relevant: tuple[str, ...]  # paper ids (Semantic Scholar corpus ids, as text)
    query_set: str  # inline_acl | inline_nonacl | manual_acl | manual_iclr
    specificity: str  # broad | specific

    @property
    def stratum(self) -> str:
        return f"{self.query_set}/{self.specificity}"


def query_id(text: str) -> str:
    """Stable id from the query text: the same query gets the same id everywhere."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def parse_query(row: Mapping[str, Any]) -> Query:
    text = clean_text(row["query"])
    return Query(
        qid=query_id(text),
        text=text,
        relevant=tuple(str(c) for c in row["corpusids"]),
        query_set=row["query_set"],
        specificity=SPECIFICITY[int(row["specificity"])],
    )


def assign_splits(strata: Mapping[str, str], dev_fraction: float = DEV_FRACTION) -> dict[str, str]:
    """Map each qid to 'dev' or 'test', stratified by `strata` (qid -> stratum).

    Within each stratum, qids are sorted (they are hashes, so the order is
    effectively random but fixed) and the first round(n * dev_fraction) go to
    dev. Stratifying keeps both splits' mix of query types close to the full
    set's, so a dev result isn't, say, mostly broad queries. Hashing instead
    of a seeded shuffle makes the split independent of row order, library
    versions and seeds.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for qid, stratum in strata.items():
        groups[stratum].append(qid)
    splits: dict[str, str] = {}
    for qids in groups.values():
        qids.sort()
        n_dev = round(len(qids) * dev_fraction)
        for i, qid in enumerate(qids):
            splits[qid] = "dev" if i < n_dev else "test"
    return splits


def paper_row(
    corpusid: int, title: str, abstract: str | None, year: float | None, doi: str | None
) -> tuple[str, str, str, date | None, str | None]:
    """(id, title, abstract, published, doi) for COPY into papers.

    LitSearch only has a year, so `published` is January 1st of that year:
    good enough for "recent vs older" breakdowns, not for day-level cutoffs.
    """
    published = date(int(year), 1, 1) if year else None
    return (str(corpusid), clean_text(title), clean_text(abstract or ""), published, doi or None)


# --- fetch (network; needs the `bench` dependency group) --------------------


def fetch(cache: Path = CACHE) -> None:
    """Download the columns we need into `cache`, unless already there."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    if (cache / "corpus.parquet").exists() and (cache / "queries.parquet").exists():
        print(f"using cached {cache}")
        return
    fs = HfFileSystem()
    root = f"datasets/{REPO}@{REVISION}"

    def read(config: str, columns: list[str]) -> pa.Table:
        tables = []
        for path in sorted(fs.glob(f"{root}/{config}/*.parquet")):
            print(f"reading {columns} from {path.rsplit('/', 2)[-2:]}", flush=True)
            with fs.open(path) as f:
                tables.append(pq.read_table(f, columns=columns))
        return pa.concat_tables(tables)

    clean = read("corpus_clean", ["corpusid", "title", "abstract"])
    meta = read("corpus_s2orc", ["corpusid", "externalids", "year"])
    ids = meta["externalids"]
    meta = pa.table(
        {
            "corpusid": meta["corpusid"],
            "year": meta["year"],
            "doi": pc.struct_field(ids, "doi"),
            # Not loaded yet; kept for haystack mode (dedup against arXiv) and
            # DBLP venues later.
            "arxiv": pc.struct_field(ids, "arxiv"),
            "dblp": pc.struct_field(ids, "dblp"),
        }
    )
    corpus = clean.join(meta, keys="corpusid", join_type="left outer")
    cache.mkdir(parents=True, exist_ok=True)
    pq.write_table(corpus, cache / "corpus.parquet")
    pq.write_table(
        read("query", ["query_set", "query", "specificity", "corpusids"]), cache / "queries.parquet"
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()
    return rows


# --- split file -------------------------------------------------------------


def check_split(splits: dict[str, str], path: Path = SPLIT_FILE) -> None:
    """Write the split on first run; afterwards fail loudly if it ever changes.

    A silently different split would make dev and test numbers from different
    runs incomparable, and could move tuned-on queries into test.
    """
    if not path.exists():
        path.write_text(json.dumps(dict(sorted(splits.items())), indent=0) + "\n")
        print(f"wrote {path}: commit it")
        return
    committed = json.loads(path.read_text())
    if committed != splits:
        changed = sum(committed.get(q) != s for q, s in splits.items())
        raise RuntimeError(
            f"split differs from {path} for {changed} queries "
            f"({len(committed)} committed, {len(splits)} now). "
            "Did the dataset revision or the split rule change?"
        )


# --- load ---------------------------------------------------------------------


def load_corpus(conn: psycopg.Connection, rows: Iterable[Mapping[str, Any]]) -> int:
    """COPY the corpus into papers, once. Reloading would drop its embeddings."""
    existing = conn.execute("SELECT count(*) FROM papers").fetchone()
    if existing and existing[0]:
        print(f"corpus already loaded ({existing[0]:,} papers), skipping")
        return int(existing[0])
    n = 0
    with conn.transaction(), conn.cursor() as cur:
        copy_sql = "COPY papers (id, title, abstract, published, doi) FROM STDIN"
        with cur.copy(copy_sql) as copy:
            for r in rows:
                copy.write_row(
                    paper_row(r["corpusid"], r["title"], r["abstract"], r["year"], r["doi"])
                )
                n += 1
    return n


def load_queries(conn: psycopg.Connection, queries: list[Query], splits: Mapping[str, str]) -> int:
    """Replace eval_queries and eval_qrels. Returns the number of labels dropped
    because the labelled paper is missing from the corpus."""
    with conn.transaction():
        conn.execute("DELETE FROM eval_queries")  # cascades to eval_qrels
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO eval_queries (qid, text, split, facets) VALUES (%s, %s, %s, %s)",
                [
                    (
                        q.qid,
                        q.text,
                        splits[q.qid],
                        Jsonb({"query_set": q.query_set, "specificity": q.specificity}),
                    )
                    for q in queries
                ],
            )
            # Insert labels through a join on papers, so a label pointing outside
            # the corpus is dropped (and counted) instead of failing the FK.
            cur.execute("CREATE TEMP TABLE stage_qrels (qid TEXT, paper_id TEXT) ON COMMIT DROP")
            with cur.copy("COPY stage_qrels FROM STDIN") as copy:
                for q in queries:
                    for paper_id in q.relevant:
                        copy.write_row((q.qid, paper_id))
            cur.execute(
                "INSERT INTO eval_qrels (qid, paper_id) SELECT DISTINCT s.qid, s.paper_id "
                "FROM stage_qrels s JOIN papers p ON p.id = s.paper_id"
            )
            inserted = cur.rowcount
    return sum(len(set(q.relevant)) for q in queries) - inserted


def main() -> None:
    fetch()
    queries = [parse_query(r) for r in _read_rows(CACHE / "queries.parquet")]
    if len({q.qid for q in queries}) != len(queries):
        raise RuntimeError("two LitSearch queries share a qid (duplicate query text?)")
    splits = assign_splits({q.qid: q.stratum for q in queries})
    check_split(splits)

    applied = migrate(SCHEMA)
    print(f"migrated {SCHEMA}: {', '.join(applied) or 'up to date'}")
    # autocommit: each `conn.transaction()` in the loaders is then a real commit.
    with connect(SCHEMA, autocommit=True) as conn:
        n_papers = load_corpus(conn, _read_rows(CACHE / "corpus.parquet"))
        dropped = load_queries(conn, queries, splits)

    n_dev = sum(s == "dev" for s in splits.values())
    print(
        f"{n_papers:,} papers, {len(queries)} queries ({n_dev} dev / {len(queries) - n_dev} test)"
    )
    print(f"labels pointing outside the corpus (dropped): {dropped}")


if __name__ == "__main__":
    main()
