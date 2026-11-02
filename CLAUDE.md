# PaperGraph — find research papers that match your interests

A literature discovery engine anyone can use. You describe a research interest
in plain language (or give a few papers you like), and it returns ranked papers
with a grounded "why this is relevant" explanation. You can then ask questions
across the papers you selected. Under the hood: hybrid retrieval, a citation
graph, reranking, and cited generation.

**The eval harness is the centerpiece.** Benchmarks with public ground truth are
used only for evaluation; they are not the product corpus. Every retrieval,
graph or reranking change has to be justified by a measured, reproducible eval
run.

The owner is learning AI engineering and DevOps. See "Working mode" below; it
matters.

## Working mode (read first)

Claude acts as a **coach on the core and a builder on the boilerplate**.

**The user writes these (Claude explains, scaffolds signatures and tests, reviews):**
- `src/papergraph/retrieval/`: chunking, BM25, dense search, hybrid fusion
- `src/papergraph/graph/`: citation-graph features and neighbor expansion
- `src/papergraph/rerank/`: reranker integration and comparison
- `src/papergraph/evals/metrics.py`: all metric implementations

For these: explain the concept and its trade-offs, propose a function signature
and failing unit tests, then let the user implement it. Review their code
critically (correctness, edge cases, complexity). Give hints before full
solutions. If the user explicitly says "just write it", do so, but explain each
part afterwards.

**Claude writes freely:** Dockerfiles, compose, Makefile, CI, config loading,
data download and parsing, DB schema and migrations, FastAPI glue, UI,
plotting and report scripts, and the eval *runner* plumbing (not the metrics).

Always explain *why* for non-obvious DevOps choices (multi-stage builds,
healthchecks, layer caching, job scheduling). Learning DevOps is an explicit
goal.

## Product features

1. **Interest search:** a free-text interest goes in, a ranked list of papers
   comes out, each with a 1–2 sentence relevance explanation that cites the abstract.
2. **Seed-paper recommendations:** the user picks papers they like, and the
   system recommends related work through the citation graph plus embeddings.
3. **Paper deep-dive Q&A:** questions over the full text of one or more selected
   papers, with paragraph-level citations or an explicit "not in these papers".
4. **Fresh papers:** a scheduled job ingests new arXiv papers daily.

## Corpus (the product data)

- **Metadata and abstracts:** arXiv (Kaggle `Cornell-University/arxiv` snapshot,
  then the arXiv API for daily updates). Start with `cs.CL` + `cs.IR` + `cs.LG`
  for development, then scale to all of `cs.*`. Each scale-up needs a
  latency and recall check.
- **Citation graph:** OpenAlex (open, CC0), joined to arXiv on DOI or arXiv ID.
  Semantic Scholar is a fallback.
- **Full text:** only on demand for deep-dives, plus a cached subset. PDFs are
  parsed with GROBID, which runs as a Docker service. Respect arXiv's rate
  limits and terms.

Two indexing levels:
- **Paper level** (one vector and a BM25 doc per paper over title + abstract)
  drives search and recommendations.
- **Paragraph level** (chunks of full text) drives deep-dive Q&A.

## Graph

Nodes are papers (plus paragraphs for full-text papers), authors, and
optionally venues and topics. Edges are typed and weighted, and stored in
Postgres:
- `cites`: paper → paper
- **co-citation** (cited together) and **bibliographic coupling** (share
  references), derived from `cites`
- `authored_by`: paper → author
- `similar`: top-k embedding neighbors (optional; ablate it)
- `next` / `in_section`: paragraph structure for full-text papers

Expansion strategies to implement and compare:
1. No expansion (baseline)
2. 1-hop expansion with weights per edge type
3. Personalized PageRank seeded by first-stage hits or by the user's seed papers

## Pipeline

```
interest query → [optional rewrite] → BM25 + dense (paper level) → RRF fusion (top-N seeds)
              → graph expansion → reranker (top-k) → explanations grounded in abstracts
seed papers   → PPR over citation graph + embedding neighbors → rerank → explanations
deep-dive Q   → paragraph retrieval within the selected papers → rerank → cited answer or abstain
```

Every stage is toggled by config, so an ablation is a YAML change, not a code change.

## Evaluation

Each benchmark is indexed by **the same pipeline** into its own namespace, so the
labels stay valid. "Haystack mode" adds the benchmark corpus to the main arXiv
corpus as distractors, which is harder and closer to real use.

| Feature | Benchmark | Ground truth | Metrics |
|---|---|---|---|
| Interest search | **LitSearch** (literature-search questions, ~64k-paper corpus) | relevant papers | Recall@k, nDCG@10, MRR |
| Interest search | **SciDocs** (BEIR version) | relevant papers | nDCG@10 |
| Seed recommendations | **Citation holdout** (built from the corpus) | a paper's held-out references | Recall@k, nDCG@k |
| Deep-dive Q&A | **QASPER** | evidence paragraphs and answers | evidence Recall@k, Answer-F1, abstention P/R |
| Explanations and answers | LLM judge | rubric | faithfulness, citation precision/recall |
| Judge calibration | ~100 human labels | your own labels | Cohen's κ |
| Ops | traces | — | p50/p95 latency per stage, tokens, $ per query |

The citation holdout must prevent leakage. For a target paper P, give a
description of P (or a subset of its references as seeds) and predict its other
references. **Remove P and every edge created after P's publication date** from
the graph before retrieving. Without this, the graph "cheats".

Rigor rules:
- Tune on dev splits only. **Test splits are used only for final reported
  numbers.** Flag anything that tunes on test.
- Report **bootstrap 95% CIs**. Compare configs with **paired** tests on the
  same queries. A difference inside the CI is not a win.
- Break results down by query type: broad vs specific interests, and recent vs
  older papers. Also break down by graph density, since sparse-citation papers
  are where the graph can't help.
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
- Embeddings to compare: `allenai/specter2` (built for scientific papers) vs
  `BAAI/bge-small-en-v1.5`
- Rerankers to compare: cross-encoder (`BAAI/bge-reranker-v2-m3`) vs LLM rerank
- LLM: Claude via the `anthropic` SDK (`claude-sonnet-5-5` generator,
  `claude-opus-5-5` judge). Key from `ANTHROPIC_API_KEY` in `.env`, never committed.
- API: FastAPI. Tracing: Arize Phoenix, self-hosted.

## Layout

```
src/papergraph/
  ingest/      arXiv metadata, OpenAlex citations, GROBID full text, chunk, embed, load
  retrieval/   bm25, dense, fusion
  graph/       edges, co-citation/coupling, expansion, PPR
  rerank/
  generate/    explanations, deep-dive answers, citations, abstention
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
- `grobid`: PDF → structured full text
- `ingest`: one-off job (profile `jobs`); also run daily for new arXiv papers
- `eval`: one-off job (profile `jobs`), mounts `reports/`
- `phoenix`: tracing UI

Model weights are cached in a named volume so containers don't re-download them.

## Commands

```
make up            # db + api + grobid + phoenix
make ingest        # corpus (CATS=cs.CL,cs.IR,cs.LG LIMIT=...)
make ingest-bench  # index a benchmark into its namespace (BENCH=litsearch|scidocs|qasper)
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
2. **Corpus ingest:** arXiv metadata and abstracts for the dev categories in
   Postgres, with paper-level embeddings.
3. **Eval harness v0:** LitSearch indexed, runner, retrieval metrics, run
   artifacts. Do this before any clever retrieval.
4. **Search baselines:** BM25, dense (SPECTER2 vs bge), hybrid RRF. First comparison table.
5. **Reranking:** cross-encoder vs LLM rerank, with the quality vs latency trade-off.
6. **Citation graph:** OpenAlex edges, 1-hop expansion, PPR, and seed
   recommendations evaluated with the leakage-safe citation holdout.
7. **Deep-dive Q&A:** GROBID full text, paragraph retrieval, QASPER evals,
   abstention.
8. **Generation quality:** grounded explanations, LLM judge, and judge calibration
   against human labels.
9. **Ops:** tracing, cost and latency reporting, daily ingest job, CI eval gate,
   haystack-mode evals at full scale.
10. **Final report:** `REPORT.md` with test-split numbers, CIs, ablations, and
    failure analysis. Public demo UI.

## Conventions

- Type hints everywhere. Keep core modules as small pure functions so they are easy to test.
- No LangChain or LlamaIndex. Build the primitives by hand to learn them.
- Claims in the README or REPORT must link to a run directory.
- Update the milestone checklist above as work lands.
