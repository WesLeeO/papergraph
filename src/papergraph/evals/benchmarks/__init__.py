"""Benchmark loaders. Each one indexes its corpus into its own schema."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Benchmark:
    schema: str  # corpus namespace in Postgres
    revision: str  # pinned dataset version: the corpus snapshot id of a run


def get_benchmark(name: str) -> Benchmark:
    # Imported here so `papergraph.evals.benchmarks` stays cheap to import.
    if name == "litsearch":
        from papergraph.evals.benchmarks import litsearch

        return Benchmark(schema=litsearch.SCHEMA, revision=litsearch.REVISION)
    raise ValueError(f"unknown benchmark: {name!r}")
