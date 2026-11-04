"""Embed papers and store the vectors. Run with `make embed [MODEL=specter2] [LIMIT=...]`.

The model (the product's, PRODUCT_MODEL, by default; or any name in MODELS)
decides the target table; see papergraph.encoders, which query embedding
shares so the vectors match.

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
from psycopg import sql

from papergraph.db import connect
from papergraph.encoders import PRODUCT_MODEL, device, get_encoder, get_model

_MISSING = """
SELECT p.id, p.title, p.abstract
FROM papers p
WHERE NOT EXISTS (SELECT 1 FROM {table} e WHERE e.paper_id = p.id)
  -- A paper with no text would get the embedding of "", identical for all of
  -- them, which dense search could then return as a block of junk.
  AND (p.title <> '' OR p.abstract <> '')
-- Newest first: a partial run (LIMIT, or stopped early) covers recent papers,
-- which matter most for search. Ordering by id alone would be oldest first,
-- since new-style arXiv ids start with YYMM. id breaks ties within a day.
ORDER BY p.published DESC, p.id
LIMIT %s
"""


def _build_index(conn: psycopg.Connection, table: str, mem: str) -> None:
    """Create the HNSW index if missing. vector_cosine_ops makes `<=>` use it."""
    # The build is fast only while the whole graph fits in maintenance_work_mem;
    # pgvector logs a notice when it doesn't. This SET lasts for this session only.
    conn.execute(f"SET maintenance_work_mem = '{mem}'")
    start = time.perf_counter()
    index = f"{table}_hnsw_idx"
    create = "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding vector_cosine_ops)"
    conn.execute(sql.SQL(create).format(sql.Identifier(index), sql.Identifier(table)))
    print(f"index {index} ready in {time.perf_counter() - start:,.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default="public", help="corpus namespace, e.g. bench_litsearch")
    parser.add_argument(
        "--model", default=PRODUCT_MODEL, help="a name in papergraph.encoders.MODELS"
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after N papers")
    parser.add_argument("--chunk", type=int, default=5000, help="papers fetched per round")
    parser.add_argument("--batch-size", type=int, default=None, help="papers per forward pass")
    parser.add_argument(
        "--index-mem", default="2GB", help="maintenance_work_mem for the HNSW build"
    )
    args = parser.parse_args()

    model = get_model(args.model)
    encoder = get_encoder(model.name)
    batch_size = args.batch_size or model.batch_size
    table = sql.Identifier(model.table)
    missing = sql.SQL(_MISSING).format(table=table)
    copy_sql = sql.SQL("COPY {} (paper_id, embedding) FROM STDIN (FORMAT BINARY)").format(table)
    index = sql.Identifier(f"{model.table}_hnsw_idx")
    drop_index = sql.SQL("DROP INDEX IF EXISTS {}").format(index)
    print(f"model {model.name} on {device()} -> {args.schema}.{model.table}")

    start = time.perf_counter()
    done = 0
    # autocommit: otherwise the first SELECT opens a transaction that lasts the
    # whole run, `conn.transaction()` only makes savepoints inside it, and an
    # interrupted run loses everything. With autocommit, each chunk commits.
    with connect(args.schema, autocommit=True) as conn:
        register_vector(conn)  # teaches psycopg to send numpy arrays as pgvector values
        while args.limit is None or done < args.limit:
            size = args.chunk if args.limit is None else min(args.chunk, args.limit - done)
            rows = conn.execute(missing, (size,)).fetchall()
            if not rows:
                break
            if done == 0:  # there is work to do: drop the index until the load is over
                conn.execute(drop_index)
            # Unit vectors (the encoders normalize), so cosine similarity = dot product.
            vectors = encoder.encode_papers([(t, a) for _, t, a in rows], batch_size)
            with conn.transaction(), conn.cursor() as cur, cur.copy(copy_sql) as copy:
                copy.set_types(["text", "vector"])
                for (paper_id, _, _), vec in zip(rows, vectors, strict=True):
                    copy.write_row((paper_id, vec))
            done += len(rows)
            rate = done / (time.perf_counter() - start)
            print(f"{done:>9,} embedded  ({rate:,.0f} papers/s)", flush=True)

        print(f"embedded {done:,} papers in {time.perf_counter() - start:,.0f}s")
        _build_index(conn, model.table, args.index_mem)


if __name__ == "__main__":
    main()
