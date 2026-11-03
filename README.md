# PaperGraph

**Find machine-learning papers that match your interests, and let a research
agent dig through them for you. Runs entirely on your own computer, for free.**

- **Search.** Describe what you're interested in, in plain language ("making
  LLM inference faster with quantization"), and get ranked papers from arXiv's
  machine-learning categories. Filter by year, venue (NeurIPS, ICLR, ACL, CVPR,
  …) and category.
- **Research agent.** Ask a research question. A local AI model plans
  searches, reads the results, searches again where something is missing, and
  writes an answer with a reading list. It may only cite papers it actually
  retrieved, and every citation is checked.

No API keys and no cloud services: the data lives in a local Postgres
database, and the AI models run on your machine.

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | macOS, Linux, or Windows with [WSL 2](https://learn.microsoft.com/windows/wsl/install) | macOS (Apple Silicon) or Linux with an NVIDIA GPU |
| RAM | 8 GB (search only) | 16 GB or more (search + research agent) |
| Free disk | ~10 GB (quick mode) | ~25 GB (all ~580k papers) |
| GPU | not required (CPU works, embedding is slower) | Apple Silicon or NVIDIA |

Software:

- [Docker Desktop](https://docs.docker.com/get-docker/), to run the database
- [uv](https://docs.astral.sh/uv/getting-started/installation/), the Python package manager (it installs Python 3.12 for you)
- `make` (preinstalled on macOS and Linux; on Windows, use it inside WSL 2)
- [Ollama](https://ollama.com/download), *only for the research agent*

## Quick start

### 1. Clone

```bash
git clone https://github.com/WesLeeO/papergraph.git
cd papergraph
```

### 2. Get the data

PaperGraph indexes the [arXiv metadata snapshot](https://www.kaggle.com/datasets/Cornell-University/arxiv)
that Cornell University publishes on Kaggle (free, CC0): titles, abstracts and
authors of every arXiv paper.

```bash
make data
```

This downloads it with the Kaggle command-line tool (a 1.75 GB zip, ~5 GB once
unzipped; about a minute on a fast connection). No Kaggle account is needed
for this public dataset.

If the download fails, either:
- **use a Kaggle API token:** create a free Kaggle account, go to
  *Settings → API → Create New Token*, save the downloaded `kaggle.json` as
  `~/.kaggle/kaggle.json`, and run `make data` again; or
- **download it in the browser** from the
  [Kaggle page](https://www.kaggle.com/datasets/Cornell-University/arxiv),
  unzip it, and move `arxiv-metadata-oai-snapshot.json` into `data/raw/`.

Either way, you should end up with `data/raw/arxiv-metadata-oai-snapshot.json`.

### 3. Set everything up

Start Docker Desktop, then run:

```bash
make setup
```

This one command:
1. creates `.env` from `.env.example`
2. starts Postgres (with pgvector) in Docker
3. installs the Python dependencies
4. loads the papers
5. builds the keyword index
6. computes the paper embeddings
7. downloads the agent's model, if Ollama is installed

By default it uses **quick mode**: only papers first published in 2026 (~100k
papers). Typical timings:

| Step | Quick mode (~100k papers) | All papers (~580k) |
|---|---|---|
| Load papers (reads the whole 5 GB file either way) | ~5–10 min | ~10–15 min |
| Keyword (BM25) index | ~2 min | ~10 min |
| Embeddings, Apple Silicon or NVIDIA GPU | ~15 min | ~1.5 h |
| Embeddings, CPU only | ~1 h | several hours |

For the full corpus, use `make setup FROM_YEAR=` (empty). To start from a
different year, e.g. 2024, use `make setup FROM_YEAR=2024`.

If something goes wrong, `make doctor` checks every prerequisite and tells
you how to fix what's missing. Setup steps can be re-run safely: the embedding
step continues where it left off.

### 4. Run it

```bash
make serve
```

Open **http://localhost:8080**. The first search takes a few extra seconds
while the models load.

### 5. The research agent (optional)

The agent needs [Ollama](https://ollama.com/download) running in the background.
Install it, then:

```bash
ollama serve          # skip if the Ollama app is already running
make agent-model      # downloads Qwen3 8B (~5 GB), once
```

Reload the page and open the *Research agent* tab. A research question takes
roughly 30 seconds to 3 minutes, depending on your hardware. You can also use
it from the terminal:

```bash
make agent Q="Which methods reduce hallucination in retrieval-augmented generation?"
```

---

## How it works

```
                      ┌─ BM25 keyword search (Postgres inverted index) ─┐
your interest ────────┤                                                 ├─ fusion (RRF) ─ reranker ─ results
                      └─ semantic search (bge-small embeddings, pgvector) ┘

research question ─ agent loop (Qwen3 8B via Ollama, running locally)
                      ├─ search_papers   the search pipeline above, with filters
                      ├─ get_paper       read a paper's full abstract
                      ├─ similar_papers  "more like this"
                      └─ search_arxiv    live arXiv search for the newest preprints
                    ─ answer + reading list (every citation checked against what the tools returned)
```

- **Keyword + semantic search, fused.** BM25 is good at exact terms (method
  names, datasets); embeddings are good at paraphrases. Reciprocal Rank Fusion
  combines the two rankings.
- **Reranking.** A small cross-encoder (`mxbai-rerank-xsmall`) reads the
  question and each of the top 20 abstracts together, and reorders them.
- **Agent.** A hand-written tool-calling loop with guardrails: a step limit,
  repeated-call detection, tool errors fed back to the model, citation
  checking, and a nudge when the model tries to answer after a single search.

Every design choice is measured on the [LitSearch](https://huggingface.co/datasets/princeton-nlp/LitSearch)
benchmark, with bootstrap confidence intervals and paired comparisons:

| Pipeline (LitSearch dev) | nDCG@10 |
|---|---|
| Postgres full-text search (baseline) | 0.352 |
| BM25 | 0.440 |
| Dense (bge-small) | 0.450 |
| Hybrid (BM25 + dense, RRF) | 0.496 |
| **Hybrid + reranker (default)** | **0.537** |

Details: [`reports/m4/`](reports/m4/README.md) (search) and
[`reports/m5/`](reports/m5/README.md) (reranking).

## Troubleshooting

| Problem | Fix |
|---|---|
| `make setup` says the data file is missing | See step 2; the file must be at `data/raw/arxiv-metadata-oai-snapshot.json` |
| `Cannot connect to the Docker daemon` | Start Docker Desktop and wait until it says it is running |
| Port 5432 is already in use | You probably run Postgres locally. In `.env`, set `DB_PORT=5433` and change `localhost:5432` to `localhost:5433` in `DATABASE_URL` |
| Port 8080 is already in use | Run `uv run --group embed uvicorn papergraph.api.main:app --port 8090` and open port 8090 |
| "Research agent unavailable" | Start Ollama (`ollama serve`) and run `make agent-model`, then reload the page |
| Embedding is very slow | Without a GPU it runs on CPU. Use quick mode, or let it run: it resumes if interrupted |
| Anything else | Run `make doctor` |

To stop everything: `Ctrl+C` in the `make serve` terminal, then `make down`
(your data is kept). `make nuke` deletes the database too.

## For developers

```bash
make check     # lint + type check + tests (no database, GPU or network needed)
make help      # every command
```

The evaluation harness (`make bench-litsearch`, `make eval`, `make compare`),
the design and the milestones are described in [`CLAUDE.md`](CLAUDE.md).

## License

[MIT](LICENSE) for the code. The arXiv metadata is released by Cornell
University under [CC0](https://creativecommons.org/publicdomain/zero/1.0/);
the models it downloads (bge-small, mxbai-rerank-xsmall, Qwen3) have their own
licenses (MIT and Apache-2.0).
