"""Per-query retrieval metrics. Averaging and confidence intervals live in stats.py.

Conventions shared by every function:
- `ranked` is the system's output, best first. If an id appears more than once,
  only its first occurrence counts (a buggy retriever must not score twice).
- A query with no relevant documents raises ValueError: it cannot be scored,
  and silently returning 0 or 1 would bias the mean. The runner drops such
  queries and logs how many.
- `k` must be >= 1. A `k` larger than len(ranked) is fine: missing ranks count
  as non-relevant.
"""

import math
from collections.abc import Collection, Mapping, Sequence


def _dedup(ranked: Sequence[str]) -> list[str]:
    """`ranked` with repeats removed, keeping each id's first (best) position."""
    return list(dict.fromkeys(ranked))  # dicts keep insertion order


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of the relevant documents found in the top k.

    |relevant ∩ top-k| / |relevant|
    """
    _check_k(k)
    if not relevant:
        raise ValueError("query has no relevant documents")
    relevant = set(relevant)  # O(1) membership, and repeats in the labels count once
    hits = sum(doc in relevant for doc in _dedup(ranked)[:k])
    return hits / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    """1 / rank of the first relevant document (ranks start at 1); 0.0 if none.

    MRR is the mean of this over queries.
    """
    if not relevant:
        raise ValueError("query has no relevant documents")
    relevant = set(relevant)
    for rank, doc in enumerate(_dedup(ranked), start=1):
        if doc in relevant:
            return 1 / rank
    return 0.0


def _dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain of gains listed in rank order (rank 1 first)."""
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))


def ndcg_at_k(ranked: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    """Normalized discounted cumulative gain at k.

    `gains` maps doc id -> relevance grade. Binary benchmarks (LitSearch) pass
    1.0 for each relevant doc; ids missing from `gains`, or with gain 0, are
    non-relevant.

    DCG@k  = sum over ranks i = 1..k of gain(doc at i) / log2(i + 1)
    IDCG@k = the same sum for the best possible ranking: all gains, sorted
             descending, truncated to k
    nDCG@k = DCG@k / IDCG@k

    Linear gain (not 2^gain - 1), to match trec_eval / BEIR numbers.
    """
    _check_k(k)
    ideal = sorted((g for g in gains.values() if g > 0), reverse=True)[:k]
    if not ideal:
        raise ValueError("query has no relevant documents")
    actual = [gains.get(doc, 0.0) for doc in _dedup(ranked)[:k]]
    return _dcg(actual) / _dcg(ideal)
