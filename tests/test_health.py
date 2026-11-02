from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from papergraph.api import main
from papergraph.config import get_settings


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    # Point at a port nothing listens on: tests must never need a real database.
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:x@127.0.0.1:1/x")
    get_settings.cache_clear()
    with TestClient(main.app) as c:
        yield c
    get_settings.cache_clear()


def test_health_is_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready_reports_unavailable_database(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["database"] == "unavailable"


def test_ready_ok_when_database_check_passes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "check_database", lambda pool: True)
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready", "database": "ok"}
