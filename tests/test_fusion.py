from typing import Any

import pytest

from papergraph.retrieval import fusion
from papergraph.retrieval.fusion import rrf


def test_single_ranking_is_unchanged() -> None:
    assert rrf([["a", "b", "c"]]) == ["a", "b", "c"]


def test_agreement_beats_a_single_first_place() -> None:
    # "b" is 2nd in both lists; "a" and "c" are 1st in one list and absent in the other.
    fused = rrf([["a", "b"], ["c", "b"]])
    assert fused[0] == "b"


def test_formula() -> None:
    # "x": rank 1 in list 1, rank 3 in list 2 -> 1/61 + 1/63
    # "y": rank 2 in both                     -> 2/62
    fused = rrf([["x", "y", "z"], ["z", "y", "x"]], k_rrf=60)
    assert 1 / 61 + 1 / 63 > 2 / 62  # so x and z (symmetric) beat y
    assert fused == ["x", "z", "y"]  # x/z tie, broken by id


def test_small_k_rrf_rewards_top_ranks() -> None:
    # With k_rrf=0, being 1st somewhere (1/1) beats being 2nd and 3rd (1/2 + 1/3).
    assert rrf([["a", "b"], ["c", "x", "b"]], k_rrf=0)[0] in {"a", "c"}
    # With a large k_rrf, appearing in both lists wins.
    assert rrf([["a", "b"], ["c", "x", "b"]], k_rrf=1000)[0] == "b"


def test_weights() -> None:
    assert rrf([["a"], ["b"]], weights=[1.0, 2.0])[0] == "b"
    with pytest.raises(ValueError):
        rrf([["a"], ["b"]], weights=[1.0])


def test_duplicates_within_a_list_count_once() -> None:
    # A repeated id must not collect score twice, nor shift later ranks.
    assert rrf([["a", "a", "b"], ["b", "a"]]) == rrf([["a", "b"], ["b", "a"]])


def test_empty_rankings() -> None:
    assert rrf([[], []]) == []


def test_search_fuses_sub_retrievers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int, dict[str, Any]]] = []

    def fake(name: str, ranking: list[str]) -> Any:
        def retrieve(conn: Any, query: str, k: int, **params: Any) -> list[str]:
            calls.append((name, k, params))
            return ranking[:k]

        return retrieve

    registry = {"one": fake("one", ["a", "b", "c"]), "two": fake("two", ["c", "b", "a"])}
    monkeypatch.setattr("papergraph.retrieval.registry.RETRIEVERS", registry)
    out = fusion.search(
        None,  # type: ignore[arg-type]
        "q",
        2,
        retrievers=[{"kind": "one", "params": {"p": 1}}, {"kind": "two"}],
        candidates=3,
    )
    # a and c: 1/61 + 1/63 each; b: 2/62. Tie between a and c broken by id.
    assert out == ["a", "c"]
    # Each sub-retriever is asked for `candidates`, not the final k.
    assert [(n, k) for n, k, _ in calls] == [("one", 3), ("two", 3)]
    assert calls[0][2] == {"p": 1}


def test_search_rejects_typos_in_sub_retrievers() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        fusion.search(None, "q", 5, retrievers=[{"kind": "bm25", "parms": {}}])  # type: ignore[arg-type]
