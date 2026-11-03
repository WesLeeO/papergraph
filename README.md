# PaperGraph

Find ML papers that match your interests, and let a research agent dig through
them for you. Hybrid retrieval (BM25 + dense), cross-encoder reranking, an
agentic RAG loop on a local open model, and a rigorous evaluation harness. See
`CLAUDE.md` for the full design and milestones.

- **Search:** describe an interest in plain language, get ranked papers from
  581k arXiv ML papers, filterable by year, venue and category.
- **Research agent:** ask a research question; a local LLM (Qwen3 via Ollama)
  plans searches, reads results, searches again, and writes an answer citing
  only papers it actually retrieved. Free: no API key, runs on your machine.

## Run the web app

Needs Docker, [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

```bash
cp .env.example .env
make up                 # Postgres + pgvector in Docker
make migrate            # create the tables
make ingest             # arXiv ML papers -> Postgres (needs data/raw/arxiv-metadata-oai-snapshot.json)
make embed              # bge-small vectors, on the Mac GPU (~1.5 h for 581k papers)
uv run python -m papergraph.retrieval.bm25   # BM25 index (~10 min)

ollama serve            # in another terminal, if Ollama isn't already running
ollama pull qwen3:8b    # ~5 GB, once

make serve              # http://localhost:8080
```

The agent also works in the terminal:

```bash
make agent Q="Which methods reduce hallucination in retrieval-augmented generation?"
```

## How it works

```
question ──► BM25 (Postgres inverted index) ──┐
         └─► dense (bge-small + pgvector HNSW) ┴─► RRF fusion ─► cross-encoder rerank (top 20) ─► papers

research question ─► agent loop (Qwen3, local)
                       ├─ search_papers   (the pipeline above, with filters)
                       ├─ get_paper / similar_papers
                       └─ search_arxiv    (live arXiv API, newest preprints)
                     ─► cited answer (citations checked against tool results)
```

Every retrieval choice is backed by an eval run on LitSearch: see
`reports/m4/` (search baselines) and `reports/m5/` (reranking).

## Development

```bash
make check        # lint + type check + tests (no database, network or GPU needed)
make eval CONFIG=configs/rerank20_mxbai_xsmall.yaml BENCH=litsearch
make compare A=reports/runs/<a> B=reports/runs/<b>
```
