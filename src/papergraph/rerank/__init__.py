"""Second-stage rerankers: reorder first-stage candidates with a stronger model.

Each one: (conn, query, paper_ids, **params) -> the same ids, best first.
"""

from collections.abc import Callable

from papergraph.rerank import cross_encoder

Reranker = Callable[..., list[str]]

RERANKERS: dict[str, Reranker] = {"cross_encoder": cross_encoder.rerank}


def get_reranker(kind: str) -> Reranker:
    try:
        return RERANKERS[kind]
    except KeyError:
        raise ValueError(f"unknown reranker {kind!r}; known: {sorted(RERANKERS)}") from None
