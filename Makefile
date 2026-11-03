.PHONY: help install up down logs ps nuke migrate ingest embed bench-litsearch eval compare serve agent psql test lint format check

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

install:  ## Install Python deps locally (incl. dev tools)
	uv sync

up:  ## Build and start the stack in the background
	docker compose up -d --build --wait

down:  ## Stop the stack (keeps the database volume)
	docker compose down

logs:  ## Follow logs from all services
	docker compose logs -f

ps:  ## Show service status and health
	docker compose ps

nuke:  ## Stop the stack AND delete the database volume
	docker compose down -v

migrate:  ## Apply pending database migrations (inside the api image)
	docker compose run --rm --build api python -m papergraph.migrate

LIMIT ?=
ingest:  ## Load arXiv ML papers into Postgres (LIMIT=5000 for a quick run)
	docker compose run --rm --build ingest python -m papergraph.ingest.load $(if $(LIMIT),--limit $(LIMIT))

MODEL ?= bge-small
embed:  ## Embed papers on the host GPU (needs `make up`; MODEL=specter2, LIMIT=5000)
	uv run --group embed python -m papergraph.ingest.embed --model $(MODEL) $(if $(LIMIT),--limit $(LIMIT))

bench-litsearch:  ## Load, index (BM25) and embed LitSearch into schema bench_litsearch (needs `make up`)
	uv run --group bench python -m papergraph.evals.benchmarks.litsearch
	uv run python -m papergraph.retrieval.bm25 --schema bench_litsearch
	uv run --group embed python -m papergraph.ingest.embed --schema bench_litsearch --model bge-small
	uv run --group embed python -m papergraph.ingest.embed --schema bench_litsearch --model specter2

CONFIG ?= configs/baseline.yaml
BENCH ?= litsearch
SPLIT ?= dev
eval:  ## Run an eval: CONFIG=configs/x.yaml BENCH=litsearch [SPLIT=dev] [LIMIT=50]
	uv run --group embed python -m papergraph.evals.runner --config $(CONFIG) --bench $(BENCH) --split $(SPLIT) $(if $(LIMIT),--limit $(LIMIT))

compare:  ## Paired diff of two runs: A=reports/runs/<a> B=reports/runs/<b>
	uv run python -m papergraph.evals.compare $(A) $(B)

serve:  ## Web UI + API on the host (Mac GPU) at http://localhost:8080; needs `make up` and `ollama serve`
	PAPERGRAPH_WARMUP=1 uv run --group embed uvicorn papergraph.api.main:app --host 127.0.0.1 --port 8080

Q ?=
agent:  ## Research agent in the terminal: make agent Q="your question"
	uv run --group embed python -m papergraph.agent "$(Q)"

psql:  ## Open a SQL shell on the database
	docker compose exec db psql -U papergraph

test:  ## Run unit tests (no Docker, network or API key needed)
	uv run pytest

lint:  ## Lint, format-check and type-check
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:  ## Auto-fix lint issues and format code
	uv run ruff check --fix .
	uv run ruff format .

check: lint test  ## What CI runs
