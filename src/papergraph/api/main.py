"""FastAPI app: health checks, paper search, the research agent, and the web UI.

    GET /                 the web UI (static page)
    GET /api/search       ranked papers for a free-text interest, with filters
    GET /api/agent        the research agent, streamed as Server-Sent Events
    GET /api/meta         filter options (venues, categories) and corpus size

Run on the host with `make serve` (uses the Mac GPU for the embedder and the
reranker); the agent needs a local Ollama server (`ollama serve`).
"""

import os
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, StreamingResponse
from psycopg_pool import ConnectionPool

from papergraph import __version__
from papergraph import search as svc
from papergraph.agent.llm import DEFAULT_MODEL, OllamaClient
from papergraph.agent.loop import Event, run_agent
from papergraph.agent.tools import Toolbox
from papergraph.db import check_database, create_pool
from papergraph.doctor import check_ollama
from papergraph.ingest.arxiv import ML_CATEGORIES
from papergraph.query import KINDS
from papergraph.retrieval.filters import Filters

STATIC = Path(__file__).parent / "static"


def _warm_up(pool: ConnectionPool) -> None:
    """Load the embedder, reranker and local LLM in the background, so the first
    user request doesn't pay ~1 minute of model loading."""
    try:
        with pool.connection() as conn:
            svc.search(conn, "warm up the embedding and reranking models", k=1)
        OllamaClient(DEFAULT_MODEL).chat([{"role": "user", "content": "Say OK."}], None)
        print("warm-up done: models loaded", flush=True)
    except Exception as e:  # warm-up is best effort
        print(f"warm-up skipped: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    pool = create_pool()
    pool.open(wait=False)
    app.state.pool = pool
    if os.environ.get("PAPERGRAPH_WARMUP") == "1":
        threading.Thread(target=_warm_up, args=(pool,), daemon=True).start()
    yield
    pool.close()


app = FastAPI(title="PaperGraph", version=__version__, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is up. Never touches dependencies."""
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready(request: Request, response: Response) -> dict[str, str]:
    """Readiness: dependencies (Postgres + pgvector) are reachable."""
    if check_database(request.app.state.pool):
        return {"status": "ready", "database": "ok"}
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "not ready", "database": "unavailable"}


@app.get("/")
def index() -> FileResponse:
    # no-cache: browsers revalidate on each visit, so an updated page shows up
    # right away instead of a stale cached copy (cheap: it's one small file).
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def make_filters(
    year_from: int | None, year_to: int | None, venues: str, categories: str
) -> Filters:
    return Filters(year_from, year_to, split_csv(venues), split_csv(categories))


Year = Annotated[int | None, Query(ge=1900, le=2100)]


@app.get("/api/search")
def api_search(
    request: Request,
    q: Annotated[str, Query(max_length=500)] = "",
    author: Annotated[str, Query(max_length=200)] = "",
    offset: Annotated[int, Query(ge=0, le=5000)] = 0,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
    year_from: Year = None,
    year_to: Year = None,
    venues: str = "",
    categories: str = "",
    skip: str = "",
    rerank: bool = True,
) -> dict[str, Any]:
    """One page of results for what the user typed.

    `q` is parsed (papergraph.query): author names, venues, years and arXiv ids
    become filters, the rest is the topic. `understood` reports what was
    parsed so the UI can show it as chips; `skip` (e.g. "author,venue") turns
    a kind of parsing off when the user removes its chip. The other parameters
    are the filter panel and take precedence. The full ranking is computed
    once and cached, so the next page (offset += limit) is cheap.
    """
    if not q.strip() and not author.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "give q, author, or both")
    if year_from and year_to and year_from > year_to:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "From year is after To year")
    skipped = [k for k in split_csv(skip) if k in KINDS]
    start = time.perf_counter()
    with request.app.state.pool.connection() as conn:
        r = svc.run_query(
            conn,
            q,
            author=author,
            filters=make_filters(year_from, year_to, venues, categories),
            skip=skipped,
            rerank=rerank,
        )
        papers = svc.fetch_papers(conn, r.ids[offset : offset + limit])
    u, end = r.understood, offset + len(papers)
    return {
        "query": q,
        "understood": {
            "topic": u.topic,
            "author": r.author_name if not author.strip() else None,
            "venues": list(u.venues),
            "venue_year": u.venue_year,
            "year_from": u.year_from,
            "year_to": u.year_to,
            "arxiv_id": u.arxiv_id,
            "spans": u.spans,
        },
        "author": r.author_name,
        "matched_authors": [{"name": n, "papers": c} for n, c in r.matched_authors],
        "skip": skipped,
        "papers": [p.to_dict() for p in papers],
        "offset": offset,
        "total": len(r.ids),
        "next_offset": end if end < len(r.ids) else None,
        "ms": round((time.perf_counter() - start) * 1000),
    }


@app.get("/api/agent")
def api_agent(
    request: Request,
    q: Annotated[str, Query(min_length=3, max_length=1000)],
    think: bool = False,
) -> StreamingResponse:
    """Stream the agent's steps as Server-Sent Events: one `data: {json}` per event."""
    pool: ConnectionPool = request.app.state.pool

    def events() -> Iterator[str]:
        try:
            with pool.connection() as conn:
                for event in run_agent(q, OllamaClient(think=think), Toolbox(conn)):
                    yield f"data: {event.to_json()}\n\n"
        except Exception as e:  # surface failures in the UI instead of a dropped stream
            yield f"data: {Event('error', {'message': f'{type(e).__name__}: {e}'}).to_json()}\n\n"
        yield 'data: {"type": "done"}\n\n'

    # no-cache + no proxy buffering: each event must reach the browser immediately.
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(events(), media_type="text/event-stream", headers=headers)


@app.get("/api/meta")
def api_meta(request: Request) -> dict[str, Any]:
    try:
        with request.app.state.pool.connection(timeout=5) as conn:
            row = conn.execute(
                "SELECT (SELECT count(*) FROM papers), (SELECT max(published) FROM papers)"
            ).fetchone()
            venues = svc.venues(conn)
    except Exception as e:
        detail = f"database unavailable: {e}"
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail) from e
    agent_ok, agent_message = check_ollama()
    return {
        "venues": [{"name": v, "count": n} for v, n in venues],
        "categories": sorted(ML_CATEGORIES),
        "papers": row[0] if row else 0,
        "latest": row[1].isoformat() if row and row[1] else None,
        # The agent needs a local Ollama server; search works without it.
        "agent": {"available": agent_ok, "message": agent_message},
    }
