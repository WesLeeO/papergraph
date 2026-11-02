import re

from papergraph.migrate import migration_files


def test_migration_files_are_numbered_and_unique() -> None:
    names = [f.name for f in migration_files()]
    assert names, "no migrations found (are they packaged?)"
    assert all(re.fullmatch(r"\d{4}_[a-z0-9_]+\.sql", n) for n in names), names
    prefixes = [n[:4] for n in names]
    assert len(prefixes) == len(set(prefixes)), "two migrations share a number"
    assert names == sorted(names)
