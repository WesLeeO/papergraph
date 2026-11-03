"""Contract for evals/stats.py. These fail until you implement it.

Bootstrap outputs are random, so most tests check properties (ordering,
determinism, widths against the textbook formula) rather than exact numbers.
"""

import math

import numpy as np
import pytest

from papergraph.evals.stats import bootstrap_ci, paired_diff_ci, paired_permutation_test

# --- bootstrap_ci -----------------------------------------------------------


def test_estimate_is_the_sample_mean() -> None:
    ci = bootstrap_ci([0.0, 1.0, 1.0, 0.0, 1.0])
    assert ci.estimate == pytest.approx(0.6)
    assert ci.low <= ci.estimate <= ci.high


def test_constant_values_give_zero_width() -> None:
    ci = bootstrap_ci([0.3] * 50)
    assert (ci.low, ci.estimate, ci.high) == pytest.approx((0.3, 0.3, 0.3))


def test_is_deterministic_for_a_seed() -> None:
    # Continuous values: with a handful of 0/1 values the resampled means take
    # so few distinct values that two seeds can give identical quantiles.
    values = np.random.default_rng(1).uniform(0, 1, 30).tolist()
    assert bootstrap_ci(values, n_resamples=500, seed=7) == bootstrap_ci(
        values, n_resamples=500, seed=7
    )
    assert bootstrap_ci(values, n_resamples=500, seed=7) != bootstrap_ci(
        values, n_resamples=500, seed=8
    )


def test_width_matches_normal_approximation() -> None:
    # 400 queries, half hits: standard error = sqrt(0.5 * 0.5 / 400) = 0.025,
    # so a 95% interval is about 2 * 1.96 * 0.025 = 0.098 wide.
    ci = bootstrap_ci([1.0, 0.0] * 200)
    assert ci.high - ci.low == pytest.approx(2 * 1.96 * math.sqrt(0.25 / 400), abs=0.01)


def test_more_queries_give_a_narrower_interval() -> None:
    small = bootstrap_ci([1.0, 0.0] * 10)
    large = bootstrap_ci([1.0, 0.0] * 1000)
    assert (large.high - large.low) < (small.high - small.low) / 5


def test_higher_confidence_is_wider() -> None:
    values = [1.0, 0.0, 0.0] * 30
    ci90 = bootstrap_ci(values, confidence=0.90)
    ci99 = bootstrap_ci(values, confidence=0.99)
    assert ci99.low <= ci90.low and ci90.high <= ci99.high


@pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5, -0.1])
def test_bad_confidence_raises(confidence: float) -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([0.0, 1.0], confidence=confidence)


def test_empty_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([])


# --- paired_diff_ci ---------------------------------------------------------


def test_identical_runs_have_zero_difference() -> None:
    a = [0.0, 1.0, 0.5, 0.2]
    ci = paired_diff_ci(a, a)
    assert (ci.low, ci.estimate, ci.high) == pytest.approx((0.0, 0.0, 0.0))


def test_direction_is_b_minus_a() -> None:
    assert paired_diff_ci([0.0, 0.0], [1.0, 1.0]).estimate == pytest.approx(1.0)


def test_constant_improvement_is_exact_despite_query_variance() -> None:
    # Queries vary wildly in difficulty, but b beats a by exactly 0.1 on each.
    # Pairing sees that: the interval collapses to 0.1.
    a = [0.0, 0.9, 0.1, 0.5, 0.3, 0.8]
    b = [x + 0.1 for x in a]
    ci = paired_diff_ci(a, b)
    assert (ci.low, ci.estimate, ci.high) == pytest.approx((0.1, 0.1, 0.1))


def test_pairing_detects_what_separate_cis_miss() -> None:
    rng = np.random.default_rng(42)
    a = rng.uniform(0, 1, 200).tolist()  # per-query difficulty dominates
    b = [x + 0.02 for x in a]  # small but consistent real improvement
    ci_a, ci_b = bootstrap_ci(a), bootstrap_ci(b)
    assert ci_b.low < ci_a.high, "separate CIs overlap: looks like noise"
    assert paired_diff_ci(a, b).low > 0, "paired CI excludes 0: it's real"


def test_paired_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        paired_diff_ci([0.0, 1.0], [0.0])
    with pytest.raises(ValueError):
        paired_diff_ci([], [])


# --- paired_permutation_test ------------------------------------------------


def test_identical_runs_have_p_one() -> None:
    a = [0.0, 1.0, 0.5]
    assert paired_permutation_test(a, a) == 1.0


def test_differences_that_cancel_have_p_one() -> None:
    # Mean difference is exactly 0: no evidence at all against "the same".
    a = [0.5] * 10
    b = [0.6, 0.4] * 5
    assert paired_permutation_test(a, b) == pytest.approx(1.0)


def test_consistent_improvement_has_small_p() -> None:
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 0.8, 50).tolist()
    b = [x + 0.2 for x in a]
    p = paired_permutation_test(a, b, n_resamples=10_000)
    assert p < 0.01
    assert p > 0, "never exactly 0: the +1 correction"


def test_p_is_symmetric_and_deterministic() -> None:
    a = [0.1, 0.4, 0.4, 0.9, 0.0, 0.3]
    b = [0.2, 0.3, 0.6, 0.9, 0.1, 0.5]
    p = paired_permutation_test(a, b, seed=3)
    assert 0 < p <= 1
    assert p == paired_permutation_test(b, a, seed=3)
    assert p == paired_permutation_test(a, b, seed=3)


def test_permutation_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        paired_permutation_test([0.0, 1.0], [0.0])
    with pytest.raises(ValueError):
        paired_permutation_test([], [])
