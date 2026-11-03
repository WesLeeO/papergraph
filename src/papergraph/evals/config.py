"""Experiment configs: one YAML per run, so an ablation is a YAML change."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    # extra="forbid": a typo like `dept: 50` fails loudly instead of silently
    # running with the default and producing a misleading result.
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetrieverConfig(_Strict):
    kind: str  # key in runner.RETRIEVERS
    params: dict[str, Any] = Field(default_factory=dict)


class RerankConfig(_Strict):
    kind: str  # key in papergraph.rerank.RERANKERS
    # First-stage candidates to rerank. 20 keeps the full quality gain at 1/5
    # of the cost of 100 on LitSearch dev (M5 depth ablation).
    top_n: int = Field(default=20, ge=1)
    params: dict[str, Any] = Field(default_factory=dict)


class RunConfig(_Strict):
    name: str = Field(pattern=r"^[a-z0-9_-]+$")  # used in the run directory name
    retriever: RetrieverConfig
    rerank: RerankConfig | None = None  # optional second stage
    depth: int = Field(default=100, ge=1)  # how many papers the retriever returns
    recall_ks: tuple[int, ...] = (10, 20, 50, 100)
    ndcg_k: int = 10


def load_config(path: Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(path.read_text()))
