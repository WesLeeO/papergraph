"""Cross-encoder reranking (default: mixedbread-ai/mxbai-rerank-xsmall-v1).

A bi-encoder (the dense retriever) embeds query and paper separately and
compares two vectors; a cross-encoder reads them TOGETHER in one forward pass
("[query] [SEP] [title + abstract]") and outputs one relevance score. Attention
runs across both texts, so it can check the paper against each part of the
question. Nothing can be precomputed, so it costs one forward pass per
(query, paper) pair: affordable for ~100 first-stage candidates, not for the
whole corpus. That is why it is a second stage.

The model is loaded lazily on first use (importing needs no torch).
"""

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import psycopg

from papergraph.encoders import device
from papergraph.ingest.text import paper_text

# Default chosen in M5 on LitSearch dev (reranking the hybrid's top 20):
# mxbai-rerank-xsmall-v1 (71M) gains +0.042 nDCG@10 over no reranking in
# ~0.43 s/query; bge-reranker-v2-m3 (568M) gains +0.061 in ~1.07 s, a gap
# that is not significant. Speed won. See reports/m5/.
MODEL = "mixedbread-ai/mxbai-rerank-xsmall-v1"


@lru_cache(maxsize=2)
def _model(name: str, max_length: int, fp16: bool) -> Any:
    # Any: sentence-transformers is only in the embed group (absent in CI).
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(name, device=device(), max_length=max_length)
    if fp16:
        model.model.half()  # ~2x faster on MPS; scores shift slightly
    return model


def order_by_scores(paper_ids: Sequence[str], scores: Sequence[float]) -> list[str]:
    """Ids sorted by score, best first; ties keep the first-stage order."""
    if len(paper_ids) != len(scores):
        raise ValueError(f"{len(paper_ids)} ids but {len(scores)} scores")
    order = sorted(range(len(paper_ids)), key=lambda i: (-scores[i], i))
    return [paper_ids[i] for i in order]


def _texts(conn: psycopg.Connection, paper_ids: Sequence[str]) -> dict[str, str]:
    rows = conn.execute(
        "SELECT id, title, abstract FROM papers WHERE id = ANY(%s)", (list(paper_ids),)
    )
    return {pid: paper_text(title, abstract) for pid, title, abstract in rows}


def rerank(
    conn: psycopg.Connection,
    query: str,
    paper_ids: Sequence[str],
    model: str = MODEL,
    batch_size: int = 8,
    max_length: int = 512,
    fp16: bool = False,
) -> list[str]:
    """Reorder `paper_ids` by cross-encoder relevance to `query`."""
    if not paper_ids:
        return []
    texts = _texts(conn, paper_ids)
    pairs = [(query, texts.get(pid, "")) for pid in paper_ids]
    scores = _model(model, max_length, fp16).predict(
        pairs, batch_size=batch_size, show_progress_bar=False
    )
    return order_by_scores(paper_ids, [float(s) for s in scores])
