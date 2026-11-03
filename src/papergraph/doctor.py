"""Check prerequisites and setup progress: `make doctor`.

Each check prints OK or what to do. Safe to run any time: it only reads.
"""

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import httpx
import psycopg

from papergraph.agent.llm import DEFAULT_HOST, DEFAULT_MODEL
from papergraph.config import get_settings

DATA_FILE = Path("data/raw/arxiv-metadata-oai-snapshot.json")
MIN_FREE_GB = 15

Check = Callable[[], tuple[bool, str]]


def check_docker() -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return (
            False,
            "Docker not found. Install Docker Desktop: https://docs.docker.com/get-docker/",
        )
    ok = subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    return (
        ok,
        "Docker is running" if ok else "Docker is installed but not running: start Docker Desktop",
    )


def check_disk() -> tuple[bool, str]:
    free_gb = shutil.disk_usage(".").free / 1e9
    ok = free_gb >= MIN_FREE_GB
    msg = f"{free_gb:.0f} GB free disk"
    return ok, msg if ok else f"{msg}; the full corpus needs ~{MIN_FREE_GB} GB (quick mode ~5 GB)"


def check_data() -> tuple[bool, str]:
    if DATA_FILE.exists():
        return True, f"arXiv snapshot found ({DATA_FILE.stat().st_size / 1e9:.1f} GB)"
    return False, f"{DATA_FILE} missing: run `make data` or see README > Get the data"


def check_database() -> tuple[bool, str]:
    try:
        with psycopg.connect(get_settings().database_url, connect_timeout=3) as conn:
            row = conn.execute(
                "SELECT (SELECT count(*) FROM papers),"
                " EXISTS (SELECT 1 FROM bm25_postings),"  # not count(*): 50M+ rows
                " (SELECT count(*) FROM paper_embeddings_bge_small)"
            ).fetchone()
    except psycopg.errors.UndefinedTable:
        return False, "Database is up but empty: run `make setup`"
    except psycopg.OperationalError:
        return (
            False,
            "Cannot reach Postgres: run `docker compose up -d db` (and check DB_PORT in .env)",
        )
    assert row is not None
    papers, postings, vectors = row
    if not papers:
        return False, "No papers loaded yet: run `make setup`"
    if not postings:
        return (
            False,
            f"{papers:,} papers, but no BM25 index: `uv run python -m papergraph.retrieval.bm25`",
        )
    if not vectors:
        return False, f"{papers:,} papers, but no embeddings: `make embed`"
    note = "" if vectors >= papers * 0.99 else f" (dense search covers the newest {vectors:,})"
    return True, f"{papers:,} papers, BM25 index, {vectors:,} embeddings{note}"


def check_accelerator() -> tuple[bool, str]:
    try:
        from papergraph.encoders import device
    except ImportError:
        return False, "ML dependencies missing: run `uv sync --group embed`"
    try:
        dev = device()
    except ImportError:
        return False, "torch not installed: run `uv sync --group embed`"
    if dev == "cpu":
        return True, "No GPU found: models run on CPU (works, embedding is slower)"
    return True, f"Models will run on {dev.upper()}"


def check_ollama() -> tuple[bool, str]:
    try:
        tags = httpx.get(f"{DEFAULT_HOST}/api/tags", timeout=2).json()
    except httpx.HTTPError:
        return False, (
            "Ollama not reachable (only the research agent needs it): install from "
            "https://ollama.com, then run `ollama serve`"
        )
    names = {m.get("name", "") for m in tags.get("models", [])}
    if DEFAULT_MODEL not in names:
        return False, f"Ollama is running but {DEFAULT_MODEL} is missing: run `make agent-model`"
    return True, f"Ollama running with {DEFAULT_MODEL}: research agent ready"


CHECKS: list[tuple[str, Check]] = [
    ("Docker", check_docker),
    ("Disk", check_disk),
    ("Data", check_data),
    ("Database", check_database),
    ("Accelerator", check_accelerator),
    ("Agent (optional)", check_ollama),
]


def main() -> None:
    failed = 0
    for name, check in CHECKS:
        try:
            ok, msg = check()
        except Exception as e:  # a broken check must not hide the others
            ok, msg = False, f"check failed: {type(e).__name__}: {e}"
        failed += not ok
        print(f"  {'OK ' if ok else '!! '} {name:<17} {msg}")
    print("\nAll good: `make serve` and open http://localhost:8080" if not failed else "")


if __name__ == "__main__":
    main()
