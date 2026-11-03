import pytest

from papergraph.db import search_path


def test_public_path() -> None:
    assert search_path("public") == "public"


def test_benchmark_schema_keeps_public_for_pgvector() -> None:
    assert search_path("bench_litsearch") == "bench_litsearch, public"


@pytest.mark.parametrize("bad", ["", "Bench", "bench-litsearch", "x; DROP TABLE papers", "1abc"])
def test_rejects_unsafe_schema_names(bad: str) -> None:
    with pytest.raises(ValueError):
        search_path(bad)
