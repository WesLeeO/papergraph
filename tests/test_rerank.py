import subprocess
import sys

import pytest
from pydantic import ValidationError

from papergraph.evals.config import RunConfig
from papergraph.evals.runner import QueryResult, merge_reranked, summarize
from papergraph.rerank import get_reranker
from papergraph.rerank.cross_encoder import order_by_scores


def test_merge_puts_reranked_head_first_and_keeps_the_tail() -> None:
    ranked = ["a", "b", "c", "d", "e"]
    assert merge_reranked(ranked, ["c", "a", "b"]) == ["c", "a", "b", "d", "e"]


def test_merge_rejects_a_reranker_that_changes_the_candidates() -> None:
    with pytest.raises(ValueError, match="permutation"):
        merge_reranked(["a", "b", "c"], ["a", "x"])


def test_order_by_scores_best_first_ties_keep_first_stage_order() -> None:
    assert order_by_scores(["a", "b", "c", "d"], [0.1, 0.9, 0.5, 0.9]) == ["b", "d", "c", "a"]


def test_order_by_scores_length_mismatch() -> None:
    with pytest.raises(ValueError):
        order_by_scores(["a"], [0.1, 0.2])


def test_rerank_config_is_optional_and_validated() -> None:
    base = {"name": "t", "retriever": {"kind": "bm25"}}
    assert RunConfig.model_validate(base).rerank is None
    cfg = RunConfig.model_validate({**base, "rerank": {"kind": "cross_encoder", "top_n": 50}})
    assert cfg.rerank is not None and cfg.rerank.top_n == 50
    with pytest.raises(ValidationError):
        RunConfig.model_validate({**base, "rerank": {"kind": "cross_encoder", "top_n": 0}})
    with pytest.raises(ValidationError):  # typo in a field name
        RunConfig.model_validate({**base, "rerank": {"kind": "cross_encoder", "topn": 50}})


def test_unknown_reranker() -> None:
    with pytest.raises(ValueError, match="unknown reranker"):
        get_reranker("llm-magic")


def test_summary_reports_latency_per_stage() -> None:
    results = [
        QueryResult(
            f"q{i}",
            {},
            [],
            {"x": 1.0},
            10.0 + i,
            {"mrr": 1.0},
            {"retrieve": 2.0 + i, "rerank": 8.0},
        )
        for i in range(4)
    ]
    s = summarize(results)
    assert s["stage_latency_ms"]["retrieve"] == {"p50": 3.0, "p95": 5.0}
    assert s["stage_latency_ms"]["rerank"] == {"p50": 8.0, "p95": 8.0}


def test_import_does_not_load_torch() -> None:
    code = "import papergraph.rerank, papergraph.evals.runner, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
