"""Embed papers with bge-small and store the vectors. Run with `make embed [LIMIT=...]`.

Runs on the host, not in Docker: Docker on macOS cannot use the Apple GPU (MPS).
Resumable: each round selects papers that have no vector yet and commits on its
own, so stopping and re-running continues where it left off.

The HNSW index is dropped before loading and built once at the end: linking
rows into the graph one insert at a time is much slower than one bulk build.
"""

import argparse
import time

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from papergraph.db import connect
from papergraph.ingest.text import paper_text

MODEL = "BAAI/bge-small-en-v1.5"
TABLE = "paper_embeddings_bge_small"
INDEX = f"{TABLE}_hnsw_idx"

_MISSING = f"""
SELECT p.id, p.title, p.abstract
FROM papers p
WHERE NOT EXISTS (SELECT 1 FROM {TABLE} e WHERE e.paper_id = p.id)
  -- A paper with no text would get the embedding of "", identical for all of
  -- them, which dense search could then return as a block of junk.
  AND (p.title <> '' OR p.abstract <> '')
-- Newest first: a partial run (LIMIT, or stopped early) covers recent papers,
-- which matter most for search. Ordering by id alone would be oldest first,
-- since new-style arXiv ids start with YYMM. id breaks ties within a day.
ORDER BY p.published DESC, p.id
LIMIT %s
"""


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _build_index(conn: psycopg.Connection, mem: str) -> None:
    """Create the HNSW index if missing. vector_cosine_ops makes `<=>` use it."""
    # The build is fast only while the whole graph fits in maintenance_work_mem;
    # pgvector logs a notice when it doesn't. This SET lasts for this session only.
    conn.execute(f"SET maintenance_work_mem = '{mem}'")
    start = time.perf_counter()
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {INDEX} ON {TABLE} USING hnsw (embedding vector_cosine_ops)"
    )
    print(f"index {INDEX} ready in {time.perf_counter() - start:,.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default="public", help="corpus namespace, e.g. bench_litsearch")
    parser.add_argument("--limit", type=int, default=None, help="stop after N papers")
    parser.add_argument("--chunk", type=int, default=5000, help="papers fetched per round")
    parser.add_argument("--batch-size", type=int, default=128, help="papers per forward pass")
    parser.add_argument(
        "--index-mem", default="2GB", help="maintenance_work_mem for the HNSW build"
    )
    args = parser.parse_args()

    device = _device()
    model = SentenceTransformer(MODEL, device=device)
    print(f"model {MODEL} on {device}")

    start = time.perf_counter()
    done = 0
    # autocommit: otherwise the first SELECT opens a transaction that lasts the
    # whole run, `conn.transaction()` only makes savepoints inside it, and an
    # interrupted run loses everything. With autocommit, each chunk commits.
    with connect(args.schema, autocommit=True) as conn:
        register_vector(conn)  # teaches psycopg to send numpy arrays as pgvector values
        while args.limit is None or done < args.limit:
            size = args.chunk if args.limit is None else min(args.chunk, args.limit - done)
            rows = conn.execute(_MISSING, (size,)).fetchall()
            if not rows:
                break
            if done == 0:  # there is work to do: drop the index until the load is over
                conn.execute(f"DROP INDEX IF EXISTS {INDEX}")
            texts = [paper_text(title, abstract) for _, title, abstract in rows]
            # normalize_embeddings=True: unit vectors, so cosine similarity = dot product.
            vectors = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True)
            with conn.transaction(), conn.cursor() as cur:
                copy_sql = f"COPY {TABLE} (paper_id, embedding) FROM STDIN WITH (FORMAT BINARY)"
                with cur.copy(copy_sql) as copy:
                    copy.set_types(["text", "vector"])
                    for (paper_id, _, _), vec in zip(rows, vectors, strict=True):
                        copy.write_row((paper_id, vec))
            done += len(rows)
            rate = done / (time.perf_counter() - start)
            print(f"{done:>9,} embedded  ({rate:,.0f} papers/s)", flush=True)

        print(f"embedded {done:,} papers in {time.perf_counter() - start:,.0f}s")
        _build_index(conn, args.index_mem)


if __name__ == "__main__":
    main()
