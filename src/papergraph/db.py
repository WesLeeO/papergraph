"""Postgres connection pool shared by the API."""

from psycopg_pool import ConnectionPool

from papergraph.config import get_settings


def create_pool() -> ConnectionPool:
    # open=False: the pool connects lazily, so the API can start (and report
    # "not ready") even while the database is still booting.
    return ConnectionPool(get_settings().database_url, min_size=1, max_size=10, open=False)


def check_database(pool: ConnectionPool) -> bool:
    """True if Postgres answers and the pgvector extension is installed."""
    try:
        with pool.connection(timeout=2) as conn:
            row = conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone()
            return row is not None
    except Exception:
        return False
