# PaperGraph

Find research papers that match your interests. Hybrid retrieval, a citation
graph, reranking, and a rigorous evaluation harness. See `CLAUDE.md` for the
full design and milestones.

## Quickstart

```bash
cp .env.example .env
make install      # local dev tools
make check        # lint + tests
make up           # start Postgres (pgvector) + API in Docker
curl localhost:8000/ready
```
