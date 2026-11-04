"""Apply SQL migrations in order. Run with `make migrate`.

Each file in `migrations/` is named `NNNN_description.sql` and runs exactly once.
Applied versions are recorded in the `schema_migrations` table, so re-running is
safe: only new files are applied. Never edit a migration that has already been
applied anywhere; add a new one instead.

`--schema bench_litsearch` builds the same tables in another schema: each
benchmark corpus gets its own namespace, indexed by the same pipeline, so its
relevance labels stay valid. Each schema tracks its own schema_migrations.
"""

import argparse
from importlib import resources
from importlib.resources.abc import Traversable

from psycopg import sql

from papergraph.db import connect

MIGRATIONS = resources.files("papergraph") / "migrations"


def migration_files() -> list[Traversable]:
    """All migration files, sorted by their numeric prefix."""
    return sorted(
        (f for f in MIGRATIONS.iterdir() if f.name.endswith(".sql")),
        key=lambda f: f.name,
    )


def migrate(schema: str = "public") -> list[str]:
    """Apply pending migrations in `schema`; return the names of those applied."""
    applied: list[str] = []
    # autocommit, so each `conn.transaction()` below is a real transaction rather
    # than a savepoint inside one transaction spanning the whole run.
    with connect(schema, autocommit=True) as conn:
        # CREATE SCHEMA before anything else: search_path silently skips a
        # schema that doesn't exist, and tables would land in public.
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
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
                # runs whole sql file
                conn.execute(f.read_text())
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (f.name,))
            applied.append(f.name)
    return applied


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default="public", help="corpus namespace to migrate")
    args = parser.parse_args()
    names = migrate(args.schema)
    print("\n".join(f"applied {n} in {args.schema}" for n in names) or "already up to date")
