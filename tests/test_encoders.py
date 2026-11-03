import subprocess
import sys

import pytest

from papergraph.encoders import MODELS, get_model


def test_registry() -> None:
    assert get_model("bge-small").dim == 384
    assert get_model("specter2").dim == 768
    assert len({m.table for m in MODELS.values()}) == len(MODELS), "one table per model"
    with pytest.raises(ValueError, match="unknown embedding model"):
        get_model("bert-huge")


def test_import_does_not_load_torch() -> None:
    code = "import papergraph.encoders, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
