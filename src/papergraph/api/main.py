"""FastAPI app. Only health endpoints for now; search routes arrive in milestone 4."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status

from papergraph import __version__
from papergraph.db import check_database, create_pool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    pool = create_pool()
    pool.open(wait=False)
    app.state.pool = pool
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
