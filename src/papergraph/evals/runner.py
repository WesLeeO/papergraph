"""Run one retrieval config over a benchmark split and record everything.

    make eval CONFIG=configs/baseline.yaml BENCH=litsearch [SPLIT=dev] [LIMIT=50]

Writes reports/runs/<UTC timestamp>_<name>/:
  config.yaml      the resolved config (defaults filled in)
  meta.json        git SHA (+ dirty flag), benchmark, split, size, dataset revision
  per_query.jsonl  one line per query: ranked ids, metrics, latency, facets
  summary.json     mean metrics, overall and per facet value, and latency p50/p95

Defaults to the dev split. The test split is for final reported numbers only,
so running on it needs an explicit SPLIT=test.
"""

import argparse
import json
import subprocess
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import yaml

from papergraph.db import connect
from papergraph.evals.baselines import pg_fts
from papergraph.evals.benchmarks import get_benchmark
from papergraph.evals.config import RunConfig, load_config
from papergraph.evals.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from papergraph.evals.stats import bootstrap_ci
from papergraph.rerank import get_reranker, merge_reranked
from papergraph.retrieval.registry import RETRIEVERS as CORE_RETRIEVERS
from papergraph.retrieval.registry import Retriever

# Config `retriever.kind` -> function: the product retrievers plus the
# eval-only reference baseline.
RETRIEVERS: dict[str, Retriever] = {"pg_fts": pg_fts, **CORE_RETRIEVERS}

REPORTS_DIR = Path("reports")
RUNS_DIR = REPORTS_DIR / "runs"


@dataclass(frozen=True)
class QueryResult:
    qid: str
    facets: dict[str, str]
    ranked: list[str]
    relevant: dict[str, float]  # paper id -> gain
    latency_ms: float  # whole pipeline
    metrics: dict[str, float] = field(default_factory=dict)
    stage_ms: dict[str, float] = field(default_factory=dict)  # per stage: retrieve, rerank


def score(ranked: Sequence[str], gains: Mapping[str, float], cfg: RunConfig) -> dict[str, float]:
    """Every metric the config asks for, for one query."""
    relevant = [doc for doc, g in gains.items() if g > 0]
    out = {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in cfg.recall_ks}
    out[f"ndcg@{cfg.ndcg_k}"] = ndcg_at_k(ranked, gains, cfg.ndcg_k)
    out["mrr"] = reciprocal_rank(ranked, relevant)
    return out


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (p in 0-100): always an observed value."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = max(1, round(p / 100 * len(ordered)))
    return ordered[rank - 1]


def _means(results: Sequence[QueryResult]) -> dict[str, float]:
    names = results[0].metrics.keys()
    return {m: sum(r.metrics[m] for r in results) / len(results) for m in names}


def _cis(results: Sequence[QueryResult]) -> dict[str, list[float]]:
    """Bootstrap 95% CI [low, high] of each metric's mean over these queries."""
    out = {}
    for m in results[0].metrics:
        ci = bootstrap_ci([r.metrics[m] for r in results])
        out[m] = [ci.low, ci.high]
    return out


def _group(results: Sequence[QueryResult]) -> dict[str, Any]:
    return {"n": len(results), "metrics": _means(results), "ci95": _cis(results)}


def summarize(results: Sequence[QueryResult]) -> dict[str, Any]:
    """Mean metrics with bootstrap 95% CIs, overall and per facet value
    (broad vs specific, ...), plus latency percentiles.

    Facet CIs are much wider than the overall one (fewer queries): read them
    before trusting a per-facet difference.
    """
    if not results:
        raise ValueError("no results to summarize")
    by_facet: dict[str, dict[str, list[QueryResult]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for facet, value in r.facets.items():
            by_facet[facet][value].append(r)
    latencies = [r.latency_ms for r in results]
    stages = sorted({stage for r in results for stage in r.stage_ms})
    return {
        **_group(results),
        "by_facet": {
            facet: {v: _group(rs) for v, rs in sorted(values.items())}
            for facet, values in sorted(by_facet.items())
        },
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        "stage_latency_ms": {
            stage: {
                "p50": percentile([r.stage_ms[stage] for r in results], 50),
                "p95": percentile([r.stage_ms[stage] for r in results], 95),
            }
            for stage in stages
        },
    }


def run_dir_name(started: datetime, name: str) -> str:
    """Sortable, unique-enough run directory name: 20261103T120501Z_baseline-fts."""
    return f"{started.astimezone(UTC):%Y%m%dT%H%M%SZ}_{name}"


def git_state(cwd: Path | None = None) -> dict[str, Any]:
    """Commit SHA and whether the working tree had uncommitted changes.

    A dirty run can't be reproduced from its SHA alone, so it is flagged, with
    the offending paths listed. reports/ is excluded: run outputs and write-ups
    are results, not code, and counting them would mark every run after the
    first one dirty.
    """

    def git(*args: str) -> str:
        out = subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)
        return out.stdout.strip()

    status = git("status", "--porcelain", "--", ".", f":(exclude){REPORTS_DIR}")
    files = [line[3:] for line in status.splitlines()]
    return {"sha": git("rev-parse", "HEAD") or None, "dirty": bool(files), "dirty_files": files}


# --- database ---------------------------------------------------------------

_QUERIES = """
SELECT q.qid, q.text, q.facets, json_object_agg(r.paper_id, r.gain)
FROM eval_queries q JOIN eval_qrels r ON r.qid = q.qid
WHERE q.split = %s
GROUP BY q.qid
ORDER BY q.qid
"""


def load_queries(
    conn: psycopg.Connection, split: str
) -> list[tuple[str, str, dict[str, str], dict[str, float]]]:
    """(qid, text, facets, gains) per query, in qid order so LIMIT is deterministic.

    The inner join skips queries with no labels: metrics can't score them.
    """
    return [tuple(row) for row in conn.execute(_QUERIES, (split,))]


def run(cfg: RunConfig, bench: str, split: str, limit: int | None) -> Path:
    benchmark = get_benchmark(bench)
    retrieve = RETRIEVERS[cfg.retriever.kind]
    rerank = get_reranker(cfg.rerank.kind) if cfg.rerank else None
    started = datetime.now(UTC)
    # Before anything is written: the run's own new directory would otherwise
    # show up in `git status` and mark every run dirty.
    git = git_state()
    results: list[QueryResult] = []
    with connect(benchmark.schema, autocommit=True) as conn:
        queries = load_queries(conn, split)[:limit]
        if not queries:
            raise RuntimeError(f"no labelled {split} queries in {benchmark.schema}")
        for i, (qid, text, facets, gains) in enumerate(queries, start=1):
            t0 = time.perf_counter()
            ranked = retrieve(conn, text, cfg.depth, **cfg.retriever.params)
            t1 = time.perf_counter()
            stage_ms = {"retrieve": (t1 - t0) * 1000}
            if rerank is not None and cfg.rerank is not None:
                top = ranked[: cfg.rerank.top_n]
                ranked = merge_reranked(ranked, rerank(conn, text, top, **cfg.rerank.params))
                stage_ms["rerank"] = (time.perf_counter() - t1) * 1000
            latency_ms = (time.perf_counter() - t0) * 1000
            metrics = score(ranked, gains, cfg)
            results.append(QueryResult(qid, facets, ranked, gains, latency_ms, metrics, stage_ms))
            if i % 50 == 0:
                print(f"{i:>5}/{len(queries)} queries", flush=True)

    summary = summarize(results)
    out = RUNS_DIR / run_dir_name(started, cfg.name)
    out.mkdir(parents=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    meta = {
        "name": cfg.name,
        "benchmark": bench,
        "split": split,
        "n_queries": len(results),
        "limit": limit,
        "corpus_schema": benchmark.schema,
        "corpus_revision": benchmark.revision,
        "git": git,
        "started_at": started.isoformat(),
        "duration_s": round((datetime.now(UTC) - started).total_seconds(), 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (out / "per_query.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=None, help="first N queries (by qid)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.retriever.kind not in RETRIEVERS:
        parser.error(f"unknown retriever {cfg.retriever.kind!r}; known: {sorted(RETRIEVERS)}")
    if cfg.rerank is not None:
        try:
            get_reranker(cfg.rerank.kind)
        except ValueError as e:
            parser.error(str(e))
    out = run(cfg, args.bench, args.split, args.limit)

    summary = json.loads((out / "summary.json").read_text())
    print(f"\n{cfg.name} on {args.bench}/{args.split} (n={summary['n']})")
    for metric, value in summary["metrics"].items():
        low, high = summary["ci95"][metric]
        print(f"  {metric:<10} {value:.3f}  [{low:.3f}, {high:.3f}]")
    lat = summary["latency_ms"]
    print(f"  latency    p50 {lat['p50']:.0f} ms, p95 {lat['p95']:.0f} ms")
    for stage, st in summary["stage_latency_ms"].items():
        print(f"    {stage:<8} p50 {st['p50']:.0f} ms, p95 {st['p95']:.0f} ms")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
