"""Apply SQL migrations in order. Run with `make migrate`.

Each file in `migrations/` is named `NNNN_description.sql` and runs exactly once.
Applied versions are recorded in the `schema_migrations` table, so re-running is
safe: only new files are applied. Never edit a migration that has already been
applied anywhere; add a new one instead.
"""

from importlib import resources
from importlib.resources.abc import Traversable

import psycopg

from papergraph.config import get_settings

MIGRATIONS = resources.files("papergraph") / "migrations"


def migration_files() -> list[Traversable]:
    """All migration files, sorted by their numeric prefix."""
    return sorted(
        (f for f in MIGRATIONS.iterdir() if f.name.endswith(".sql")),
        key=lambda f: f.name,
    )


def migrate(database_url: str) -> list[str]:
    """Apply pending migrations; return the names of those applied."""
    applied: list[str] = []
    with psycopg.connect(database_url) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version TEXT PRIMARY KEY,"
            " applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        for f in migration_files():
            if f.name in done:
                continue
            # One transaction per file: a failing migration leaves no half-applied schema.
            with conn.transaction():
                conn.execute(f.read_text())
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (f.name,))
            applied.append(f.name)
    return applied


if __name__ == "__main__":
    names = migrate(get_settings().database_url)
    print("\n".join(f"applied {n}" for n in names) or "already up to date")
