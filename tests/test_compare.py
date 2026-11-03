import json
from pathlib import Path
from typing import Any

import pytest

from papergraph.evals.compare import MetricDiff, check_comparable, compare, load_run

META = {"benchmark": "litsearch", "split": "dev", "corpus_revision": "abc", "git": {"sha": "x"}}


def _write_run(
    tmp: Path, name: str, scores: dict[str, float], meta: dict[str, Any] | None = None
) -> Path:
    path = tmp / name
    path.mkdir()
    (path / "meta.json").write_text(json.dumps(meta or META))
    with (path / "per_query.jsonl").open("w") as f:
        for i, (qid, mrr) in enumerate(sorted(scores.items())):
            facets = {"specificity": "broad" if i % 2 else "specific"}
            f.write(json.dumps({"qid": qid, "facets": facets, "metrics": {"mrr": mrr}}) + "\n")
    return path


def test_identical_runs_show_no_difference(tmp_path: Path) -> None:
    scores = {f"q{i}": i / 10 for i in range(10)}
    a = load_run(_write_run(tmp_path, "a", scores))
    b = load_run(_write_run(tmp_path, "b", scores))
    (row,) = compare(a, b)["overall"]
    assert row.diff == pytest.approx(0.0)
    assert row.verdict == "no difference"
    assert row.p_value == 1.0


def test_consistent_gain_is_b_better(tmp_path: Path) -> None:
    a_scores = {f"q{i:02}": (i % 7) / 10 for i in range(40)}
    b_scores = {q: v + 0.2 for q, v in a_scores.items()}
    result = compare(
        load_run(_write_run(tmp_path, "a", a_scores)),
        load_run(_write_run(tmp_path, "b", b_scores)),
    )
    (row,) = result["overall"]
    assert row.diff == pytest.approx(0.2)
    assert row.verdict == "B better"
    assert row.p_value is not None and row.p_value < 0.01
    # Per-facet rows exist, without p-values.
    broad = result["by_facet"]["specificity"]["broad"][0]
    assert broad.p_value is None and broad.diff == pytest.approx(0.2)


def test_refuses_different_splits(tmp_path: Path) -> None:
    a = load_run(_write_run(tmp_path, "a", {"q1": 1.0}))
    b = load_run(_write_run(tmp_path, "b", {"q1": 1.0}, {**META, "split": "test"}))
    with pytest.raises(ValueError, match="split"):
        check_comparable(a, b)


def test_refuses_different_query_sets(tmp_path: Path) -> None:
    # e.g. one run used LIMIT=50: pairing on a subset would silently change the question.
    a = load_run(_write_run(tmp_path, "a", {"q1": 1.0, "q2": 0.0}))
    b = load_run(_write_run(tmp_path, "b", {"q1": 1.0}))
    with pytest.raises(ValueError, match="different queries"):
        check_comparable(a, b)


@pytest.mark.parametrize(
    ("low", "high", "verdict"),
    [(0.01, 0.1, "B better"), (-0.1, -0.01, "A better"), (-0.05, 0.08, "no difference")],
)
def test_verdict_needs_whole_ci_on_one_side(low: float, high: float, verdict: str) -> None:
    assert MetricDiff("mrr", 0.3, 0.33, 0.03, low, high, None).verdict == verdict
