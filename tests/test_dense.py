import subprocess
import sys

import pytest

from papergraph.retrieval.dense import (
    MODELS,
    effective_ef_search,
    get_model,
    query_text,
    vector_literal,
)

BGE = MODELS["bge-small"]


def test_query_instruction_is_prefixed() -> None:
    q = "  sparse attention "
    assert query_text(q, BGE, True) == BGE.query_instruction + "sparse attention"
    assert query_text(q, BGE, False) == "sparse attention"


def test_vector_literal_round_trips_floats() -> None:
    lit = vector_literal([0.1, -2.5, 1e-08])
    assert lit == "[0.1,-2.5,1e-08]"
    assert [float(x) for x in lit[1:-1].split(",")] == [0.1, -2.5, 1e-08]


@pytest.mark.parametrize(("k", "ef", "expected"), [(100, 40, 100), (10, 40, 40), (100, 200, 200)])
def test_ef_search_never_below_k(k: int, ef: int, expected: int) -> None:
    # pgvector returns at most ef_search rows: ef < k would silently truncate.
    assert effective_ef_search(k, ef) == expected


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="unknown dense model"):
        get_model("bert-huge")
    assert get_model("bge-small") is BGE


def test_import_does_not_load_torch() -> None:
    # The runner imports this module; CI and the api image have no torch.
    code = "import papergraph.retrieval.dense, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
