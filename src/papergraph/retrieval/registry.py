"""Retriever lookup by config name (`retriever.kind`)."""

from collections.abc import Callable

from papergraph.retrieval import bm25, dense, fusion

Retriever = Callable[..., list[str]]  # (conn, query, k, **params) -> paper ids, best first

RETRIEVERS: dict[str, Retriever] = {
    "bm25": bm25.search,
    "dense": dense.search,
    "hybrid": fusion.search,
}


def get_retriever(kind: str) -> Retriever:
    try:
        return RETRIEVERS[kind]
    except KeyError:
        raise ValueError(f"unknown retriever {kind!r}; known: {sorted(RETRIEVERS)}") from None
