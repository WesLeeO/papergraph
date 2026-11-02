.PHONY: help install up down logs ps nuke test lint format check

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
