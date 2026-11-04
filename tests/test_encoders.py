import re
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


def test_sentence_transformer_models_name_a_repo() -> None:
    for m in MODELS.values():
        if m.name != "specter2":  # loaded through adapters, not sentence-transformers
            assert m.hf_name, f"{m.name} has no Hugging Face repo"


def test_migrations_create_every_model_table() -> None:
    from papergraph.migrate import migration_files

    sql = "\n".join(f.read_text() for f in migration_files())
    for m in MODELS.values():
        # The table's own column, not just any VECTOR(dim) elsewhere in the file.
        created = re.search(rf"CREATE TABLE {m.table} \((.*?)\);", sql, re.DOTALL)
        assert created, f"no migration creates {m.table}"
        assert f"VECTOR({m.dim})" in created.group(1), f"{m.table} is not VECTOR({m.dim})"


def test_import_does_not_load_torch() -> None:
    code = "import papergraph.encoders, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
