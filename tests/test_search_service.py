"""Search service pieces that need no database: the ranking cache, name guards."""

import threading
import time
from typing import Any

import pytest

from papergraph import search as svc
from papergraph.search import _TTLCache, find_authors


def test_cache_computes_once_per_key() -> None:
    cache, calls = _TTLCache(), []
    for _ in range(3):
        assert cache.get_or_compute("k", lambda: calls.append(1) or [1, 2]) == [1, 2]
    assert len(calls) == 1


def test_cache_entries_expire() -> None:
    cache, calls = _TTLCache(ttl_s=0.05), []
    cache.get_or_compute("k", lambda: calls.append(1))
    time.sleep(0.08)
    cache.get_or_compute("k", lambda: calls.append(1))
    assert len(calls) == 2


def test_cache_evicts_least_recently_used() -> None:
    cache = _TTLCache(max_entries=2)
    cache.get_or_compute("a", lambda: "A")
    cache.get_or_compute("b", lambda: "B")
    cache.get_or_compute("a", lambda: "A")  # touch a: b is now the oldest
    cache.get_or_compute("c", lambda: "C")
    calls: list[str] = []
    cache.get_or_compute("a", lambda: calls.append("a") or "A")  # still cached
    cache.get_or_compute("b", lambda: calls.append("b") or "B")  # was evicted
    assert calls == ["b"]


def test_cache_is_safe_under_concurrent_use() -> None:
    cache, errors = _TTLCache(max_entries=8), []

    def worker(i: int) -> None:
        try:
            for j in range(200):
                key = (i, j % 16)
                assert cache.get_or_compute(key, lambda key=key: key) == key
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


class RecordingConn:
    """Captures the parameters find_authors sends to the database."""

    def __init__(self) -> None:
        self.params: list[Any] = []

    def execute(self, query: str, params: Any = None) -> list[Any]:
        self.params.append(params)
        return []


@pytest.mark.parametrize("name", ["", "   ", "a", "%", "_", "a b", "%%%%", "-- ''"])
def test_too_short_names_never_reach_the_database(name: str) -> None:
    conn = RecordingConn()
    assert find_authors(conn, name) == []  # type: ignore[arg-type]
    assert conn.params == []


@pytest.mark.parametrize(
    ("name", "pattern"),
    [("100%", "%100\\%%"), ("snake_case", "%snake\\_case%"), ("back\\slash", "%back\\\\slash%")],
)
def test_like_wildcards_in_names_are_escaped(name: str, pattern: str) -> None:
    conn = RecordingConn()
    find_authors(conn, name)  # type: ignore[arg-type]
    assert conn.params == [(pattern,)]


def test_empty_query_ranks_nothing() -> None:
    assert svc.ranked_ids(None, "   ") == []  # type: ignore[arg-type]
