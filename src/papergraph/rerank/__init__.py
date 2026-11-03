"""Second-stage rerankers: reorder first-stage candidates with a stronger model.

Each one: (conn, query, paper_ids, **params) -> the same ids, best first.
"""

from collections.abc import Callable, Sequence

from papergraph.rerank import cross_encoder

Reranker = Callable[..., list[str]]

RERANKERS: dict[str, Reranker] = {"cross_encoder": cross_encoder.rerank}


def get_reranker(kind: str) -> Reranker:
    try:
        return RERANKERS[kind]
    except KeyError:
        raise ValueError(f"unknown reranker {kind!r}; known: {sorted(RERANKERS)}") from None


def merge_reranked(ranked: Sequence[str], reranked_top: Sequence[str]) -> list[str]:
    """The reranked head, then the rest of the first-stage list in its order.

    The reranker reorders only the first len(reranked_top) candidates; papers
    below that keep their first-stage ranks, so Recall@100 stays comparable.
    """
    head = set(reranked_top)
    if head != set(ranked[: len(reranked_top)]):
        raise ValueError("reranker must return a permutation of the top candidates")
    return [*reranked_top, *(d for d in ranked if d not in head)]
