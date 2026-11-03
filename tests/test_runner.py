from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from papergraph.evals.config import RunConfig, load_config
from papergraph.evals.runner import (
    RETRIEVERS,
    QueryResult,
    percentile,
    run_dir_name,
    score,
    summarize,
)

CFG = RunConfig.model_validate(
    {"name": "t", "retriever": {"kind": "pg_fts"}, "recall_ks": [1, 10], "ndcg_k": 10}
)


def _result(qid: str, mrr: float, latency: float, **facets: str) -> QueryResult:
    return QueryResult(qid, facets, [], {"x": 1.0}, latency, {"mrr": mrr})


def test_score_reports_configured_metrics() -> None:
    m = score(["a", "b"], {"b": 1.0}, CFG)
    assert m.keys() == {"recall@1", "recall@10", "ndcg@10", "mrr"}
    assert m["recall@1"] == 0.0
    assert m["recall@10"] == 1.0
    assert m["mrr"] == 0.5


def test_score_ignores_zero_gain_labels_for_recall() -> None:
    # A grade-0 qrel is "judged not relevant": it must not count as missed.
    assert score(["b"], {"b": 1.0, "a": 0.0}, CFG)["recall@10"] == 1.0


def test_summarize_means_and_facets() -> None:
    results = [
        _result("q1", 1.0, 10, specificity="broad"),
        _result("q2", 0.0, 20, specificity="specific"),
        _result("q3", 0.5, 30, specificity="specific"),
    ]
    s = summarize(results)
    assert s["n"] == 3
    assert s["metrics"]["mrr"] == pytest.approx(0.5)
    low, high = s["ci95"]["mrr"]
    assert low <= s["metrics"]["mrr"] <= high
    spec = s["by_facet"]["specificity"]
    assert spec["broad"] == {"n": 1, "metrics": {"mrr": 1.0}, "ci95": {"mrr": [1.0, 1.0]}}
    assert spec["specific"]["n"] == 2
    assert spec["specific"]["metrics"]["mrr"] == pytest.approx(0.25)
    assert s["latency_ms"] == {"p50": 20, "p95": 30}


def test_summarize_empty_raises() -> None:
    with pytest.raises(ValueError):
        summarize([])


@pytest.mark.parametrize(("p", "expected"), [(0, 1), (50, 2), (95, 4), (100, 4)])
def test_percentile_nearest_rank(p: float, expected: float) -> None:
    assert percentile([4, 1, 3, 2], p) == expected


def test_run_dir_name_is_utc() -> None:
    paris = timezone(timedelta(hours=1))
    started = datetime(2026, 11, 3, 13, 5, 1, tzinfo=paris)
    assert run_dir_name(started, "baseline-fts") == "20261103T120501Z_baseline-fts"
    assert run_dir_name(started.astimezone(UTC), "x").startswith("20261103T120501Z")


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"name": "t", "retriever": {"kind": "pg_fts"}, "dept": 50})


def test_config_rejects_unsafe_name() -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"name": "../escape", "retriever": {"kind": "pg_fts"}})


def test_committed_configs_load_and_use_known_retrievers() -> None:
    paths = sorted(Path("configs").glob("*.yaml"))
    assert paths
    for path in paths:
        assert load_config(path).retriever.kind in RETRIEVERS, path


def _git(repo: Path, *args: str) -> None:
    import subprocess

    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_git_state_ignores_run_outputs_but_not_code(tmp_path: Path) -> None:
    from papergraph.evals.runner import git_state

    _git(tmp_path, "init", "-q")
    identity = ["-c", "user.email=t@t", "-c", "user.name=t"]
    _git(tmp_path, *identity, "commit", "-q", "--allow-empty", "-m", "x")
    assert git_state(tmp_path)["dirty"] is False

    # Earlier runs' outputs are untracked files, but not code changes.
    run = tmp_path / "reports" / "runs" / "20261103T000000Z_x"
    run.mkdir(parents=True)
    (run / "summary.json").write_text("{}")
    assert git_state(tmp_path)["dirty"] is False

    (tmp_path / "retriever.py").write_text("print('changed')")
    assert git_state(tmp_path)["dirty"] is True
