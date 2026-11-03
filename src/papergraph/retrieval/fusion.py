"""Hybrid retrieval: run several retrievers and merge their rankings with RRF.

Reciprocal Rank Fusion (Cormack et al., 2009):

    score(d) = sum over retrievers r of  weight_r / (k_rrf + rank_r(d))

with ranks starting at 1, and nothing added by a retriever that didn't return
d. It uses ranks only, never raw scores: BM25 scores and cosine similarities
live on unrelated scales, and RRF needs no calibration between them. k_rrf
(60 in the paper) flattens the curve, so a paper ranked well by several
retrievers beats one ranked first by a single retriever.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import psycopg

K_RRF = 60


def rrf(
    rankings: Sequence[Sequence[str]],
    *,
    k_rrf: int = K_RRF,
    weights: Sequence[float] | None = None,
) -> list[str]:
    """Fuse ranked id lists into one, best first.

    Within a list, only an id's first occurrence counts. Ties are broken by
    id, so the output is deterministic.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"{len(weights)} weights for {len(rankings)} rankings")
    if k_rrf < 0:
        raise ValueError(f"k_rrf must be >= 0, got {k_rrf}")
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        seen: set[str] = set()
        rank = 0
        for doc in ranking:
            if doc in seen:
                continue
            seen.add(doc)
            rank += 1
            scores[doc] = scores.get(doc, 0.0) + weight / (k_rrf + rank)
    return sorted(scores, key=lambda d: (-scores[d], d))


def search(
    conn: psycopg.Connection,
    query: str,
    k: int,
    retrievers: Sequence[Mapping[str, Any]] = (),
    k_rrf: int = K_RRF,
    candidates: int = 100,
) -> list[str]:
    """Top-k by RRF over sub-retrievers, each asked for `candidates` papers.

    `retrievers` comes from the config, e.g.
        [{"kind": "bm25"}, {"kind": "dense", "params": {...}, "weight": 1.0}]
    """
    # Imported here: the registry imports this module.
    from papergraph.retrieval.registry import get_retriever

    if not retrievers:
        raise ValueError("hybrid needs at least one sub-retriever")
    rankings, weights = [], []
    for spec in retrievers:
        unknown = set(spec) - {"kind", "params", "weight"}
        if unknown:
            raise ValueError(f"unknown keys in hybrid sub-retriever: {sorted(unknown)}")
        retrieve = get_retriever(spec["kind"])
        rankings.append(retrieve(conn, query, max(k, candidates), **spec.get("params", {})))
        weights.append(float(spec.get("weight", 1.0)))
    return rrf(rankings, k_rrf=k_rrf, weights=weights)[:k]
