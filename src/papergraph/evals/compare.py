"""Paired comparison of two eval runs. Run with `make compare A=<run_dir> B=<run_dir>`.

Both runs must cover the same benchmark, split, corpus revision and queries:
the comparison pairs each query's score in A with its score in B, which
cancels out how hard each query is (see stats.paired_diff_ci).

Verdict per metric: B is better only if the whole 95% CI of mean(B - A) is
above 0. A difference whose CI contains 0 is reported as "no difference",
however large the point estimate looks.
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from papergraph.evals.stats import paired_diff_ci, paired_permutation_test

# Fields that must match for two runs to be comparable.
_MUST_MATCH = ("benchmark", "split", "corpus_revision")


@dataclass(frozen=True)
class Run:
    path: Path
    meta: dict[str, Any]
    per_query: dict[str, dict[str, Any]]  # qid -> {"metrics": ..., "facets": ...}


@dataclass(frozen=True)
class MetricDiff:
    metric: str
    a: float  # mean in A
    b: float  # mean in B
    diff: float  # mean(B - A)
    low: float
    high: float
    p_value: float | None  # overall rows only; per-facet rows skip it

    @property
    def verdict(self) -> str:
        if self.low > 0:
            return "B better"
        if self.high < 0:
            return "A better"
        return "no difference"


def load_run(path: Path) -> Run:
    meta = json.loads((path / "meta.json").read_text())
    per_query = {}
    with (path / "per_query.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            per_query[row["qid"]] = {"metrics": row["metrics"], "facets": row["facets"]}
    return Run(path, meta, per_query)


def check_comparable(a: Run, b: Run) -> None:
    """Raise ValueError if pairing A and B query by query would be meaningless."""
    for key in _MUST_MATCH:
        if a.meta.get(key) != b.meta.get(key):
            raise ValueError(f"runs differ in {key}: {a.meta.get(key)!r} vs {b.meta.get(key)!r}")
    if a.per_query.keys() != b.per_query.keys():
        only_a = len(a.per_query.keys() - b.per_query.keys())
        only_b = len(b.per_query.keys() - a.per_query.keys())
        raise ValueError(
            f"runs scored different queries ({only_a} only in A, {only_b} only in B); "
            "re-run both with the same split and LIMIT"
        )


def diff_metrics(
    a: Mapping[str, Mapping[str, float]],
    b: Mapping[str, Mapping[str, float]],
    qids: Sequence[str],
    *,
    with_p: bool = True,
) -> list[MetricDiff]:
    """Paired difference for every metric both runs report, over `qids`.

    `a` and `b` map qid -> {metric: value}.
    """
    shared = [m for m in a[qids[0]] if m in b[qids[0]]]
    out = []
    for m in shared:
        xs = [a[q][m] for q in qids]
        ys = [b[q][m] for q in qids]
        ci = paired_diff_ci(xs, ys)
        out.append(
            MetricDiff(
                metric=m,
                a=sum(xs) / len(xs),
                b=sum(ys) / len(ys),
                diff=ci.estimate,
                low=ci.low,
                high=ci.high,
                p_value=paired_permutation_test(xs, ys) if with_p else None,
            )
        )
    return out


def compare(a: Run, b: Run) -> dict[str, Any]:
    """Overall paired diffs (with p-values) and per-facet diffs (CIs only).

    Per-facet rows have no p-values on purpose: testing every facet value of
    every metric is dozens of tests, and some would look "significant" by
    chance alone.
    """
    check_comparable(a, b)
    qids = sorted(a.per_query)
    am = {q: a.per_query[q]["metrics"] for q in qids}
    bm = {q: b.per_query[q]["metrics"] for q in qids}

    facets: dict[str, dict[str, list[str]]] = {}
    for q in qids:
        for facet, value in a.per_query[q]["facets"].items():
            facets.setdefault(facet, {}).setdefault(value, []).append(q)

    return {
        "n": len(qids),
        "overall": diff_metrics(am, bm, qids),
        "by_facet": {
            facet: {
                value: diff_metrics(am, bm, fq, with_p=False)
                for value, fq in sorted(values.items())
            }
            for facet, values in sorted(facets.items())
        },
    }


def _table(rows: list[MetricDiff], indent: str = "  ") -> list[str]:
    lines = []
    for r in rows:
        p = f"  p={r.p_value:.4f}" if r.p_value is not None else ""
        lines.append(
            f"{indent}{r.metric:<10} {r.a:.3f} -> {r.b:.3f}  "
            f"diff {r.diff:+.3f} [{r.low:+.3f}, {r.high:+.3f}]{p}  {r.verdict}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("a", type=Path, help="baseline run directory")
    parser.add_argument("b", type=Path, help="candidate run directory")
    args = parser.parse_args()

    a, b = load_run(args.a), load_run(args.b)
    try:
        result = compare(a, b)
    except ValueError as e:
        parser.error(str(e))

    print(f"A: {a.path.name}  (git {str(a.meta['git']['sha'])[:7]})")
    print(f"B: {b.path.name}  (git {str(b.meta['git']['sha'])[:7]})")
    print(f"{a.meta['benchmark']}/{a.meta['split']}, {result['n']} paired queries\n")
    print("\n".join(_table(result["overall"])))
    for facet, values in result["by_facet"].items():
        for value, rows in values.items():
            n = sum(1 for q in a.per_query.values() if q["facets"].get(facet) == value)
            print(f"\n  {facet}={value} (n={n}, CIs only)")
            print(
                "\n".join(
                    _table([r for r in rows if r.metric in ("recall@10", "ndcg@10", "mrr")], "    ")
                )
            )


if __name__ == "__main__":
    main()
