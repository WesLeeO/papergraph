"""Uncertainty for eval results: is a number (or a difference) real or noise?

Inputs are per-query metric values (one float per query, e.g. each query's
recall@10). Everything is seeded: the same inputs give the same output, so
reports are reproducible.

Use numpy, vectorized: draw all resample indices at once as an
(n_resamples, n) array with `rng.integers`, and take means along axis 1.
A Python loop over 10,000 resamples works but is ~100x slower.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Interval:
    estimate: float  # the statistic on the original data (NOT the resample mean)
    low: float
    high: float


def _as_array(values: Sequence[float]) -> NDArray[np.float64]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("need a non-empty 1-D sequence of per-query values")
    return x


def _differences(a: Sequence[float], b: Sequence[float]) -> NDArray[np.float64]:
    """Per-query b - a. Both runs must score the same queries, in the same order."""
    if len(a) != len(b):
        raise ValueError(f"runs have different lengths: {len(a)} vs {len(b)}")
    d: NDArray[np.float64] = _as_array(b) - _as_array(a)
    return d


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap confidence interval for the mean of `values`.

    1. Resample n values from `values` WITH replacement, n_resamples times.
    2. Take the mean of each resample.
    3. The interval is the (1 - confidence) / 2 and (1 + confidence) / 2
       quantiles of those means (2.5% and 97.5% for 95%).

    Raises ValueError for empty `values` or a confidence outside (0, 1).
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    x = _as_array(values)
    rng = np.random.default_rng(seed)  # private generator: no global side effects
    # Row r holds the indices of resample r: n draws with replacement from 0..n-1.
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    means = x[idx].mean(axis=1)  # fancy indexing -> (n_resamples, n) -> one mean per row
    alpha = 1 - confidence
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(estimate=float(x.mean()), low=float(low), high=float(high))


def paired_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Bootstrap CI for mean(b - a), where a[i] and b[i] are the same query.

    Paired: resample QUERIES (indices), keeping each query's a and b together,
    i.e. bootstrap the per-query differences. This cancels out how hard each
    query is, which is most of the variance, so it detects much smaller real
    differences than comparing two separate CIs.

    Positive = b is better. Raises ValueError if lengths differ or are 0.
    """
    # Bootstrapping the differences IS resampling queries with a and b kept
    # together: index i of d carries both a[i] and b[i].
    return bootstrap_ci(
        _differences(a, b).tolist(), n_resamples=n_resamples, confidence=confidence, seed=seed
    )


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided p-value for "a and b perform the same" (sign-flip test).

    If a and b were interchangeable, each query's difference d = b - a would
    be as likely to be +d as -d. So:
    1. observed = |mean(d)|
    2. n_resamples times: flip the sign of each d independently at random
       (each with probability 1/2) and take |mean|.
    3. p = (count of flipped |mean| >= observed, plus 1) / (n_resamples + 1)

    The +1s count the observed assignment itself as one of the permutations,
    so p is never exactly 0 (a finite test can't prove impossibility).
    Raises ValueError if lengths differ or are 0.
    """
    d = _differences(a, b)
    rng = np.random.default_rng(seed)
    observed = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_resamples, d.size))
    flipped = np.abs((signs * d).mean(axis=1))
    # Tolerance: an exact tie (e.g. all d == 0) must count as ">=", but float
    # rounding in the two means can make equal values differ in the last bit.
    extreme = np.count_nonzero(flipped >= observed - 1e-12)
    return float((extreme + 1) / (n_resamples + 1))
