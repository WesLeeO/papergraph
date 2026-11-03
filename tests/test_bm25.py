import math

import pytest

from papergraph.retrieval.bm25 import idf, score_docs, term_score

# term -> (tf_title, tf_body)
DOCS = {
    "short_match": {"attent": (0, 1), "spars": (0, 1)},
    "long_match": {"attent": (0, 1), "spars": (0, 1), "model": (0, 20)},
    "repeats": {"attent": (0, 8), "model": (0, 2)},
    "title_hit": {"attent": (1, 0), "model": (0, 1)},
    "no_match": {"model": (0, 3)},
}


def test_idf_rare_terms_weigh_more() -> None:
    assert idf(1, 1000) > idf(100, 1000) > idf(900, 1000)


def test_idf_never_negative() -> None:
    # The classic Robertson idf goes negative for terms in > half the corpus.
    assert idf(1000, 1000) > 0


def test_term_score_saturates() -> None:
    # Going from 1 to 2 occurrences adds far more than going from 20 to 21.
    s = [term_score(tf, 10, 1000, 100, 100, 1.2, 0.75) for tf in (1, 2, 20, 21)]
    assert s[1] - s[0] > 10 * (s[3] - s[2])
    # And never exceeds idf * (k1 + 1), the limit as tf -> infinity.
    assert term_score(10_000, 10, 1000, 100, 100, 1.2, 0.75) < idf(10, 1000) * 2.2


def test_term_score_matches_formula() -> None:
    tf, df, n, dl, avg, k1, b = 3, 5, 50, 80, 100, 1.2, 0.75
    expected = math.log(1 + (n - df + 0.5) / (df + 0.5)) * tf * (k1 + 1)
    expected /= tf + k1 * (1 - b + b * dl / avg)
    assert term_score(tf, df, n, dl, avg, k1, b) == pytest.approx(expected)


def test_zero_tf_scores_zero() -> None:
    assert term_score(0, 5, 50, 80, 100, 1.2, 0.75) == 0.0


def test_only_matching_docs_are_scored() -> None:
    assert "no_match" not in score_docs(["attent", "spars"], DOCS)


def test_shorter_doc_wins_on_equal_matches() -> None:
    scores = score_docs(["attent", "spars"], DOCS)
    assert scores["short_match"] > scores["long_match"]


def test_b_zero_ignores_length() -> None:
    scores = score_docs(["attent", "spars"], DOCS, b=0.0)
    assert scores["short_match"] == pytest.approx(scores["long_match"])


def test_title_boost_lifts_title_matches() -> None:
    plain = score_docs(["attent"], DOCS)
    boosted = score_docs(["attent"], DOCS, title_boost=3.0)
    rank = lambda s: sorted(s, key=s.__getitem__, reverse=True).index("title_hit")  # noqa: E731
    assert rank(boosted) < rank(plain)


def test_duplicate_query_terms_count_once() -> None:
    assert score_docs(["attent", "attent"], DOCS) == score_docs(["attent"], DOCS)
