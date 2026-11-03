"""Web API edge cases, with the database and models replaced by fakes."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient

from papergraph import search as svc
from papergraph.agent.llm import LLMError
from papergraph.api import main
from papergraph.config import get_settings
from papergraph.retrieval.filters import Filters
from papergraph.search import Paper

IDS = [f"2401.{i:05d}" for i in range(45)]  # 45 ranked results for any query


def paper(pid: str) -> Paper:
    return Paper(pid, f"Title {pid}", "Abstract.", ["A. Author"], "2024-01-01", None, None, [])


class FakePool:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    @contextmanager
    def connection(self, timeout: float | None = None) -> Iterator[object]:
        if self.fail:
            raise RuntimeError("connection refused")
        yield object()


@pytest.fixture
def calls() -> dict[str, Any]:
    return {}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, calls: dict[str, Any]) -> Iterator[TestClient]:
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:x@127.0.0.1:1/x")
    get_settings.cache_clear()

    def ranked_ids(
        conn: object, q: str, filters: Filters | None = None, rerank: bool = True
    ) -> list[str]:
        calls["search"] = (q, filters)
        return IDS

    def author_ranked_ids(
        conn: object, name: str, query: str | None = None, filters: Filters | None = None
    ) -> tuple[list[tuple[str, int]], list[str]]:
        calls["author"] = (name, query, filters)
        return ([("Ada Lovelace", 3)], IDS[:3]) if "lovelace" in name.lower() else ([], [])

    monkeypatch.setattr(svc, "ranked_ids", ranked_ids)
    monkeypatch.setattr(svc, "author_ranked_ids", author_ranked_ids)
    monkeypatch.setattr(svc, "fetch_papers", lambda conn, ids: [paper(i) for i in ids])
    with TestClient(main.app) as c:
        main.app.state.pool = FakePool()
        yield c
    get_settings.cache_clear()


# --- input validation ----------------------------------------------------------


@pytest.mark.parametrize(
    "params",
    [
        {},  # neither q nor author
        {"q": "   "},  # whitespace only
        {"q": "x" * 501},  # too long
        {"q": "rag", "year_from": 99999},  # would overflow make_date in SQL
        {"q": "rag", "year_from": 1500},
        {"q": "rag", "year_from": 2025, "year_to": 2020},  # inverted range
        {"q": "rag", "limit": 0},
        {"q": "rag", "limit": 51},
        {"q": "rag", "offset": -1},
        {"q": "rag", "year_from": "abc"},
    ],
)
def test_invalid_search_requests_are_rejected(client: TestClient, params: dict[str, Any]) -> None:
    assert client.get("/api/search", params=params).status_code == 422


# --- paging ----------------------------------------------------------------------


def test_first_page(client: TestClient) -> None:
    body = client.get("/api/search", params={"q": "rag", "limit": 20}).json()
    assert [p["id"] for p in body["papers"]] == IDS[:20]
    assert body["total"] == 45 and body["offset"] == 0 and body["next_offset"] == 20


def test_last_page_has_no_next(client: TestClient) -> None:
    body = client.get("/api/search", params={"q": "rag", "offset": 40, "limit": 20}).json()
    assert [p["id"] for p in body["papers"]] == IDS[40:]
    assert body["next_offset"] is None


def test_offset_past_the_end_is_empty_not_an_error(client: TestClient) -> None:
    body = client.get("/api/search", params={"q": "rag", "offset": 1000}).json()
    assert body["papers"] == [] and body["next_offset"] is None and body["total"] == 45


def test_pages_tile_the_ranking_without_gaps(client: TestClient) -> None:
    seen, offset = [], 0
    while offset is not None:
        body = client.get("/api/search", params={"q": "rag", "offset": offset, "limit": 7}).json()
        seen += [p["id"] for p in body["papers"]]
        offset = body["next_offset"]
    assert seen == IDS


# --- filters and the author path ----------------------------------------------------


def test_filters_reach_the_search_service(client: TestClient, calls: dict[str, Any]) -> None:
    params = {"q": "rag", "year_from": 2021, "venues": "ICLR, NeurIPS,", "categories": "cs.CL"}
    client.get("/api/search", params=params)
    assert calls["search"] == ("rag", Filters(2021, None, ("ICLR", "NeurIPS"), ("cs.CL",)))


def test_author_search(client: TestClient, calls: dict[str, Any]) -> None:
    body = client.get("/api/search", params={"author": "Ada Lovelace"}).json()
    assert body["matched_authors"] == [{"name": "Ada Lovelace", "papers": 3}]
    assert len(body["papers"]) == 3 and body["next_offset"] is None
    assert calls["author"][1] is None  # no topic: newest first


def test_author_with_topic(client: TestClient, calls: dict[str, Any]) -> None:
    client.get("/api/search", params={"author": "Ada Lovelace", "q": "engines"})
    assert calls["author"][:2] == ("Ada Lovelace", "engines")


def test_unknown_author_is_an_empty_result(client: TestClient) -> None:
    body = client.get("/api/search", params={"author": "Nobody Atall"}).json()
    assert body["matched_authors"] == [] and body["papers"] == [] and body["total"] == 0


# --- other endpoints ------------------------------------------------------------------


def test_index_is_served_without_caching(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200 and "PaperGraph" in resp.text
    assert resp.headers["cache-control"] == "no-cache"


def test_meta_reports_database_outage_as_503(client: TestClient) -> None:
    main.app.state.pool = FakePool(fail=True)
    assert client.get("/api/meta").status_code == 503


@pytest.mark.parametrize("q", ["", "ab", "x" * 1001])
def test_agent_rejects_bad_questions(client: TestClient, q: str) -> None:
    assert client.get("/api/agent", params={"q": q}).status_code == 422


def test_agent_failure_is_streamed_as_an_error_event(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DownModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

        def chat(self, messages: Any, tools: Any) -> Any:
            raise LLMError("cannot reach Ollama")

    monkeypatch.setattr(main, "OllamaClient", DownModel)
    resp = client.get("/api/agent", params={"q": "what is retrieval-augmented generation"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = [line for line in resp.text.splitlines() if line.startswith("data: ")]
    assert '"type": "error"' in events[0] and "cannot reach Ollama" in events[0]
    assert events[-1] == 'data: {"type": "done"}'
