"""Postgres connections: a pool for the API, single connections for batch jobs."""

import re

import psycopg
from psycopg_pool import ConnectionPool

from papergraph.config import get_settings


def create_pool() -> ConnectionPool:
    # open=False: the pool connects lazily, so the API can start (and report
    # "not ready") even while the database is still booting.
    # autocommit: each statement commits on its own, so a `conn.transaction()`
    # in the retrievers is a real transaction and SET LOCAL settings never leak
    # into the next request that reuses the connection.
    return ConnectionPool(
        get_settings().database_url,
        min_size=1,
        max_size=10,
        open=False,
        kwargs={"autocommit": True},
    )


def check_database(pool: ConnectionPool) -> bool:
    """True if Postgres answers and the pgvector extension is installed."""
    try:
        with pool.connection(timeout=2) as conn:
            row = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
            return row is not None
    except Exception:
        return False


# Schema names are spliced into SQL (search_path cannot be a bind parameter), so
# only plain lowercase identifiers are accepted.
_SCHEMA_RE = re.compile(r"[a-z_][a-z0-9_]*")


def search_path(schema: str) -> str:
    """The search_path for a corpus namespace: its schema first, then public.

    Unqualified table names resolve to the first schema, so the same SQL reads
    `bench_litsearch.papers` or `public.papers` depending on the connection.
    public stays on the path for the pgvector extension's `vector` type.
    """
    if not _SCHEMA_RE.fullmatch(schema):
        raise ValueError(f"invalid schema name: {schema!r}")
    return "public" if schema == "public" else f"{schema}, public"


def connect(schema: str = "public", *, autocommit: bool = False) -> psycopg.Connection:
    """A connection whose unqualified table names resolve inside `schema`."""
    return psycopg.connect(
        get_settings().database_url,
        autocommit=autocommit,
        options=f"-c search_path={search_path(schema).replace(' ', '')}",
    )
