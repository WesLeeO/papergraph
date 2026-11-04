# PaperGraph — find ML papers that match your interests

Describe a research interest in plain language and get ranked ML papers,
filterable by venue and year, each with a grounded "why this matches you"
explanation. Under the hood: hybrid retrieval, a lightweight paper graph,
reranking, and LLM explanations.

**The eval harness is the centerpiece.** Benchmarks with public ground truth are
used only for evaluation; they are not the product corpus. Every retrieval,
graph or reranking change has to be justified by a measured, reproducible eval
run.

The owner is learning AI engineering and DevOps. See "Working mode" below; it
matters.

## Working mode (read first)

**Claude writes the code for every milestone**, core modules included
(retrieval, graph, reranking, metrics, stats), with tests. Changed from
"coach on the core" on 2026-11-03 at the owner's request.

Learning is still the goal, so after writing core logic, explain the concept,
the trade-offs and each non-obvious decision. When the owner writes code
themselves, fix only real bugs and keep their structure and style; suggest
refactors instead of applying them.

Always explain *why* for non-obvious DevOps choices (multi-stage builds,
healthchecks, layer caching, job scheduling). Learning DevOps is an explicit
goal.

## Product (core scope)

1. **Interest search:** a free-text interest goes in, a ranked list of papers
   comes out. Filters: venue (e.g. "ICLR/ICML/NeurIPS only"), year, category.
2. **Grounded explanations:** 1–2 sentences per top result on why it matches,
   citing only the abstract.

3. **Research agent (agentic RAG):** a research question goes in; a local open
   model (Qwen3 via Ollama, free) plans searches over the index, reads results,
   searches again, and returns an answer citing only papers it retrieved.
   Tools: local search (with filters), paper lookup, similar papers, live arXiv
   API for preprints newer than the snapshot. Agent loop written by hand.

Deferred (not core): seed-paper recommendations, daily arXiv updates.
Cut: full-text deep-dive Q&A, citation graph.

## Corpus

- **Scope: ML broadly.** Every arXiv paper with at least one of `cs.LG`,
  `stat.ML`, `cs.CL`, `cs.CV`, `cs.NE`, `cs.IR` (~581k papers in the
  Nov 2026 snapshot). Standalone `cs.AI` is excluded: it is mostly non-ML
  (logic, planning); ML papers in `cs.AI` are almost always cross-listed.
- **Source:** arXiv metadata snapshot from Kaggle (`Cornell-University/arxiv`)
  at `data/raw/arxiv-metadata-oai-snapshot.json`. JSON Lines, ~5 GB, ~3.2M
  papers: always stream it, never load it whole. Develop with `LIMIT`, then
  scale up; each scale-up needs a latency and recall check.
- **Venues** (the feature the owner cares most about). Preprints stay in the
  corpus; venue is a **filter** and a candidate **ranking signal**. Sources, in
  order of rollout:
  1. Regex over arXiv `comments`/`journal-ref` during ingest (~16% coverage).
     Skip "submitted to", "under review" and workshop mentions.
  2. DBLP full XML dump, matched to arXiv papers by normalized title.
     Measure coverage and spot-check match precision by hand.
  3. Semantic Scholar, only if an API key is obtained (journals like *Nature*).
  Normalized venue names: `NeurIPS`, `ICML`, `ICLR`, `CVPR`, `ICCV`, `ECCV`,
  `ACL`, `EMNLP`, `NAACL`, `AAAI`, `IJCAI`, `KDD`, `SIGIR`, `TMLR`, `JMLR`,
  `Nature`, `Nature Machine Intelligence`.

## Graph (lightweight, no external data)

Nodes are papers. Edges (typed, weighted, stored in Postgres):
- `same_author`: papers sharing an author (from arXiv `authors_parsed`)
- `similar`: top-k embedding neighbors
- `same_entity`: shared method or dataset. **Experiment only**, on a subset
  first, because extracting entities from 581k abstracts is expensive.

Expansion strategies to compare: none (baseline), 1-hop with per-edge-type
weights, and Personalized PageRank seeded by first-stage hits.

## Pipeline

```
interest query → BM25 + dense → RRF fusion (top-N seeds) → [graph expansion]
              → [venue/year filters] → reranker (top-k) → grounded explanations
```

Every stage is toggled by config, so an ablation is a YAML change, not a code change.

## Evaluation

| What | Benchmark | Metrics |
|---|---|---|
| Interest search | **LitSearch** (literature-search questions over ML/NLP papers) | Recall@k, nDCG@10, MRR |
| Interest search | **SciDocs** (BEIR version) | nDCG@10 |
| Graph | "Author's next paper": given an author's earlier papers, find their later ones (graph must contain only papers published before the cutoff) | Recall@k |
| Explanations | LLM judge | faithfulness to the abstract, relevance |
| Judge calibration | ~100 human labels | Cohen's κ |
| Ops | traces | p50/p95 latency per stage, tokens, $ per query |

Each benchmark is indexed by **the same pipeline** into its own namespace, so the
labels stay valid. "Haystack mode" adds the benchmark corpus to the main arXiv
corpus as distractors, which is harder and closer to real use.

Rigor rules:
- Tune on dev splits only. **Test splits are used only for final reported
  numbers.** Flag anything that tunes on test.
- Report **bootstrap 95% CIs**. Compare configs with **paired** tests on the
  same queries. A difference inside the CI is not a win.
- Break results down by query type (broad vs specific, recent vs older papers).
- Every run writes `reports/runs/<timestamp>_<name>/` with the resolved config,
  git SHA, benchmark, split and size, corpus snapshot id, per-query results
  (JSONL), and a summary JSON.
- Cache LLM and embedding calls on disk, keyed by (model, input hash), so
  reruns are cheap and deterministic. Use temperature 0 for generation and judging.
- The judge model must be different from, or stronger than, the generator.
  Judge prompts are versioned in `prompts/`, and the version is logged per run.

## Stack

- Python 3.12, `uv` for dependencies, `ruff` + `mypy` + `pytest`
- Postgres 16 + `pgvector` with HNSW indexes for vectors, full text, and graph
  edges. One DB on purpose; revisit only if the evals or latency demand it.
- Embeddings: product `Qwen/Qwen3-Embedding-0.6B` (M4b); `BAAI/bge-small-en-v1.5`
  and `allenai/specter2` kept for comparison
- Reranker: none in the product (M4b). `mixedbread-ai/mxbai-rerank-xsmall-v1`
  and `BAAI/bge-reranker-v2-m3` stay available to eval configs
- LLM: Claude via the `anthropic` SDK (`claude-sonnet-5-5` generator,
  `claude-opus-5-5` judge). Key from `ANTHROPIC_API_KEY` in `.env`, never committed.
- API: FastAPI. Tracing: Arize Phoenix, self-hosted.

## Layout

```
src/papergraph/
  ingest/      arXiv parsing + loading, venue extraction, embeddings
  migrations/  numbered SQL files, applied by migrate.py
  retrieval/   bm25, dense, fusion
  graph/       edges, expansion, PPR
  rerank/
  generate/    grounded explanations
  evals/       runner.py, metrics.py, judge.py, stats.py, compare.py, benchmarks/
  api/         FastAPI app
configs/       experiment YAMLs (baseline.yaml, hybrid.yaml, graph_ppr.yaml, ...)
prompts/       versioned generator and judge prompts
data/          raw and cache (gitignored)
reports/runs/  eval outputs (summaries committed, JSONL gitignored)
docker/        Dockerfiles
tests/
```

## Docker / DevOps

`docker compose` services:
- `db`: pgvector/pgvector:pg16, healthcheck, named volume
- `api`: multi-stage build, non-root user, `depends_on` db being healthy
- `ingest`: one-off job (profile `jobs`), mounts `data/` read-only
- `eval`: one-off job (profile `jobs`), mounts `reports/`
- `phoenix`: tracing UI

Docker on macOS cannot use the Apple GPU, so bulk embedding runs on the host
(PyTorch MPS) and writes to the same Postgres; the Docker path stays CPU-only.

## Commands

```
make up            # db + api
make migrate       # apply pending SQL migrations
make psql          # SQL shell on the database
make ingest        # arXiv ML corpus into Postgres (LIMIT=...)
make eval CONFIG=configs/baseline.yaml BENCH=litsearch [LIMIT=200]
make compare A=<run_dir> B=<run_dir>   # paired diff + CIs
make test          # pytest (no network, no API key needed)
make lint          # ruff + mypy
```

CI (GitHub Actions) runs lint and unit tests, builds the images, and runs a
**retrieval-only eval smoke test** (a small LitSearch subset, no LLM calls). The
smoke test fails if Recall@10 drops below the committed baseline threshold.

## Milestones

A milestone is done only when its eval or test evidence exists.

1. ✅ **Infra skeleton:** compose (db + api), Makefile, healthchecks, CI lint and tests.
2. ✅ **Corpus ingest:** arXiv ML papers in Postgres, regex venues, paper-level embeddings.
   581,162 papers, 12.3% with a regex venue. bge-small: all 581,162 embedded
   (checked 2026-11-04; `make embed` resumes), ~112 papers/s on MPS.
3. ✅ **Eval harness v0:** LitSearch indexed, runner, retrieval metrics, run
   artifacts. Do this before any clever retrieval. LitSearch in schema
   `bench_litsearch` (64,183 papers, 597 queries, hashed 198 dev / 399 test
   split), bootstrap CIs, `make compare` with paired tests. Reference baseline:
   `configs/baseline.yaml` (Postgres FTS), dev Recall@10 0.459 [0.389, 0.529].
4. ✅ **Search baselines:** BM25, dense (bge vs SPECTER2), hybrid RRF. First comparison table.
   See `reports/m4/README.md`. Dev winner: hybrid RRF (BM25 + bge-small),
   nDCG@10 0.496 [0.436, 0.556] vs 0.352 for the FTS baseline (paired +0.143
   [+0.103, +0.186]); Recall@100 0.813. SPECTER2 lost to bge-small (−0.198
   nDCG@10 dense). BM25 defaults kept; RRF weights not tuned.
4b. **Stronger embedder** (2026-11-04, dev evidence in `reports/m4b/README.md`):
   Qwen3-Embedding-0.6B vs bge-small: +0.113 nDCG@10 dense, +0.064 in the
   hybrid. With a Qwen3 first stage, rerankers add nothing significant.
   New product: BM25 (max_df 0.2) + Qwen3 (RRF weight 2), no reranker:
   nDCG@10 0.567, Recall@10–100 significantly above the old product, ~1/3 the
   latency. Left: full-corpus Qwen3 embedding (started 2026-11-04, ~18 h).
5. ✅ **Reranking:** cross-encoder vs LLM rerank, with the quality vs latency trade-off.
   See `reports/m5/README.md`. Cross-encoder on the hybrid: +0.06 nDCG@10,
   mostly on specific questions; depth 20 keeps the full gain at 1/5 of the
   cost. Default mxbai-rerank-xsmall @20: nDCG@10 0.537 (+0.042) in 0.43 s
   (bge-reranker-v2-m3 @20: 0.556 in 1.07 s, gap not significant). LLM rerank
   skipped: API usage is billed separately from the Claude plan.
6. **Venues v2:** DBLP title matching with measured coverage; venue as a ranking signal.
7. **Graph:** author and kNN edges, 1-hop and PPR expansion; entity edges as an experiment.
8. **Explanations:** grounded explanations, LLM judge, judge calibration.
9. **Ops:** search API + UI, tracing, cost and latency reporting, CI eval gate.
   Done (2026-11-03): web UI + API (`make serve`) with search and a streaming
   research agent; BM25 query-term pruning (max_df 0.1) for the 581k corpus.
   Left: tracing, cost/latency reporting, CI eval gate, API in Docker.
9b. **Research agent eval:** agent vs the single-shot pipeline on LitSearch dev
   (Recall of its cited papers, steps, latency); local Qwen3 vs larger models.
10. **Final report:** `REPORT.md` with test-split numbers, CIs, ablations, and
    failure analysis.

## Conventions

- Type hints everywhere. Keep core modules as small pure functions so they are easy to test.
- No LangChain or LlamaIndex. Build the primitives by hand to learn them.
- Claims in the README or REPORT must link to a run directory.
- Update the milestone checklist above as work lands.
