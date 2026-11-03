"""Dense retrieval: embed the query, return the papers with the nearest vectors.

Papers were embedded offline by papergraph.ingest.embed (title + abstract,
normalized). At query time we embed the query with the same model and ask
pgvector for the nearest vectors by cosine distance (`<=>`), served by the
HNSW index: approximate, but milliseconds instead of a scan of every vector.

Knobs (all in the config's retriever.params):
- query_instruction: BGE models were trained with a prefix on short queries
  ("Represent this sentence for searching relevant passages: "). Whether it
  helps on LitSearch's long questions is an empirical question.
- ef_search: how many candidates HNSW keeps while searching. Higher = closer
  to exact search, slower. pgvector returns at most ef_search rows, so it is
  always raised to at least k.
- exact: skip the index and compare against every vector. Slow, but it
  measures how much recall the approximate index loses.

The model is loaded lazily on first use: importing this module (as the eval
runner and the tests do) doesn't need torch.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import psycopg
from psycopg import sql

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class DenseModel:
    hf_name: str
    table: str  # vectors written by papergraph.ingest.embed
    query_instruction: str  # prefix the model expects on retrieval queries ("" = none)


MODELS = {
    "bge-small": DenseModel(
        hf_name="BAAI/bge-small-en-v1.5",
        table="paper_embeddings_bge_small",
        query_instruction="Represent this sentence for searching relevant passages: ",
    ),
}


def get_model(name: str) -> DenseModel:
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"unknown dense model {name!r}; known: {sorted(MODELS)}") from None


def query_text(query: str, model: DenseModel, use_instruction: bool) -> str:
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


@lru_cache(maxsize=4)
def _encoder(hf_name: str) -> "SentenceTransformer":
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(hf_name, device=device)


def embed_query(text: str, model: DenseModel) -> list[float]:
    # normalize_embeddings=True, like the paper vectors: cosine = dot product.
    vec = _encoder(model.hf_name).encode(text, normalize_embeddings=True)
    return [float(x) for x in vec]


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
    vec = vector_literal(embed_query(query_text(query, m, query_instruction), m))
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
