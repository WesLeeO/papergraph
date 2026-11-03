"""Contract for evals/metrics.py. These fail until you implement it.

Expected values are written as formulas, not decimals, so you can check each
one by hand against the docstrings.
"""

import math

import pytest

from papergraph.evals.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

RANKED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "e", "z"}  # "z" is relevant but never retrieved


# --- recall@k ---------------------------------------------------------------


@pytest.mark.parametrize(("k", "expected"), [(1, 0.0), (2, 1 / 3), (5, 2 / 3)])
def test_recall_counts_hits_in_top_k(k: int, expected: float) -> None:
    assert recall_at_k(RANKED, RELEVANT, k) == pytest.approx(expected)


def test_recall_k_beyond_list_length() -> None:
    assert recall_at_k(RANKED, RELEVANT, 100) == pytest.approx(2 / 3)


def test_recall_ignores_duplicate_ids() -> None:
    # Without dedup this would be 3/2: a recall above 1.
    assert recall_at_k(["b", "b", "b"], {"b", "e"}, 3) == pytest.approx(1 / 2)


def test_recall_empty_ranking_is_zero() -> None:
    assert recall_at_k([], RELEVANT, 10) == 0.0


# --- reciprocal rank --------------------------------------------------------


def test_rr_uses_first_relevant_rank() -> None:
    assert reciprocal_rank(RANKED, RELEVANT) == pytest.approx(1 / 2)


def test_rr_top_hit_is_one() -> None:
    assert reciprocal_rank(["e", "a"], RELEVANT) == 1.0


def test_rr_no_hit_is_zero() -> None:
    assert reciprocal_rank(["a", "c"], RELEVANT) == 0.0


# --- nDCG@k -----------------------------------------------------------------


def test_ndcg_binary() -> None:
    gains = {"b": 1.0, "e": 1.0, "z": 1.0}
    dcg = 1 / math.log2(3) + 1 / math.log2(6)  # hits at ranks 2 and 5
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)  # 3 hits at ranks 1-3
    assert ndcg_at_k(RANKED, gains, 5) == pytest.approx(dcg / idcg)


def test_ndcg_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(["b", "e", "a"], {"b": 1.0, "e": 1.0}, 10) == pytest.approx(1.0)


def test_ndcg_graded_order_matters() -> None:
    gains = {"a": 2.0, "b": 1.0}
    dcg = 1 / math.log2(2) + 2 / math.log2(3)  # weaker doc ranked first
    idcg = 2 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(["b", "a"], gains, 10) == pytest.approx(dcg / idcg)


def test_ndcg_ideal_is_truncated_to_k() -> None:
    # 3 relevant docs but k=1: the best possible @1 is one hit, so this is perfect.
    assert ndcg_at_k(["b"], {"b": 1.0, "e": 1.0, "z": 1.0}, 1) == pytest.approx(1.0)


def test_ndcg_zero_gain_is_not_relevant() -> None:
    # BEIR qrels list some docs with grade 0; they must not count as hits.
    assert ndcg_at_k(["a", "b"], {"a": 0.0, "b": 1.0}, 10) == pytest.approx(1 / math.log2(3))


def test_ndcg_ignores_duplicate_ids() -> None:
    assert ndcg_at_k(["b", "b"], {"b": 1.0, "e": 1.0}, 2) == pytest.approx(
        1 / (1 + 1 / math.log2(3))
    )


# --- shared contract --------------------------------------------------------


def test_no_relevant_docs_raises() -> None:
    with pytest.raises(ValueError):
        recall_at_k(RANKED, set(), 10)
    with pytest.raises(ValueError):
        reciprocal_rank(RANKED, set())
    with pytest.raises(ValueError):
        ndcg_at_k(RANKED, {}, 10)
    with pytest.raises(ValueError):
        ndcg_at_k(RANKED, {"a": 0.0}, 10)  # only zero gains: nothing relevant


@pytest.mark.parametrize("k", [0, -1])
def test_bad_k_raises(k: int) -> None:
    with pytest.raises(ValueError):
        recall_at_k(RANKED, RELEVANT, k)
    with pytest.raises(ValueError):
        ndcg_at_k(RANKED, {"b": 1.0}, k)
