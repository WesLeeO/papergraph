# syntax=docker/dockerfile:1

# ---- Stage 1: build the virtualenv with uv ----
# Build tools and the uv cache live only in this stage; the final image just
# copies the finished .venv, so it stays small and has no build toolchain.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install dependencies BEFORE copying source code. Docker caches each layer, so
# editing a .py file reuses this (slow) layer; only lockfile changes rebuild it.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
# --no-editable: COPY papergraph's code into .venv instead of linking to /app/src.
# Stage 2 only takes .venv, so a link would point to a src/ that isn't there.
# --mount=type=cache: uv's download cache lives on the host and is reused across
# builds, so rebuilds don't re-download packages. It is never saved into the image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


# ---- Stage 2: minimal runtime image ----
FROM python:3.12-slim AS runtime

# Run as an unprivileged user: a compromised process can't write system files.
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000

# The slim image has no curl, so the healthcheck uses Python's stdlib.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "papergraph.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
