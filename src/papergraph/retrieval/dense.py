"""Dense retrieval: embed the query, return the papers with the nearest vectors.

Papers were embedded offline by papergraph.ingest.embed (title + abstract,
normalized). At query time we embed the query with the same model and ask
pgvector for the nearest vectors by cosine distance (`<=>`), served by the
HNSW index: approximate, but milliseconds instead of a scan of every vector.

Knobs (all in the config's retriever.params):
- model: bge-small or specter2 (papergraph.encoders); paper vectors must
  have been built with the same model (`make embed MODEL=...`).
- query_instruction: BGE models were trained with a prefix on short queries
  ("Represent this sentence for searching relevant passages: "). Whether it
  helps on LitSearch's long questions is an empirical question. No effect for
  models without an instruction (specter2 uses a query adapter instead).
- ef_search: how many candidates HNSW keeps while searching. Higher = closer
  to exact search, slower. pgvector returns at most ef_search rows, so it is
  always raised to at least k.
- exact: skip the index and compare against every vector. Slow, but it
  measures how much recall the approximate index loses.

The model is loaded lazily on first use: importing this module (as the eval
runner and the tests do) doesn't need torch.
"""

import psycopg
from psycopg import sql

from papergraph.encoders import EmbeddingModel, get_encoder, get_model


def query_text(query: str, model: EmbeddingModel, use_instruction: bool) -> str:
    """The exact string embedded for a query."""
    query = query.strip()
    return model.query_instruction + query if use_instruction else query


def vector_literal(values: "list[float]") -> str:
    """pgvector's text format, '[0.1,0.2,...]'. Sent as text and cast with
    ::vector, so the connection needs no pgvector type registration."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def effective_ef_search(k: int, ef_search: int) -> int:
    """pgvector's HNSW returns at most ef_search rows: asking for k=100 with the
    default ef_search=40 silently returns 40. Never search with fewer than k."""
    return max(k, ef_search)


def search(
    conn: psycopg.Connection,
    query: str,
    k: int,
    model: str = "bge-small",
    query_instruction: bool = True,
    ef_search: int = 100,
    exact: bool = False,
) -> list[str]:
    """Top-k paper ids by cosine similarity to the query embedding."""
    if not query.strip():
        return []
    m = get_model(model)
    vec = vector_literal(get_encoder(m.name).encode_query(query_text(query, m, query_instruction)))
    statement = sql.SQL(
        "SELECT paper_id FROM {table} ORDER BY embedding <=> %s::vector LIMIT %s"
    ).format(table=sql.Identifier(m.table))
    # SET LOCAL lasts until the end of this transaction only, so these
    # settings never leak into other queries on the same connection.
    with conn.transaction():
        # SET takes no bind parameters; the int() makes the formatting safe.
        conn.execute(f"SET LOCAL hnsw.ef_search = {int(effective_ef_search(k, ef_search))}")
        if exact:
            conn.execute("SET LOCAL enable_indexscan = off")
        return [row[0] for row in conn.execute(statement, (vec, k))]
