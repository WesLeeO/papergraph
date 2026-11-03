"""Embedding models, shared by offline paper embedding (ingest.embed) and
online query embedding (retrieval.dense). Papers and queries must go through
the same code, or their vectors silently stop being comparable.

Models are loaded lazily: importing this module needs no torch.

- bge-small (BAAI/bge-small-en-v1.5): general-purpose, 384 dims, via
  sentence-transformers. Paper text = "title\\nabstract" (ingest.text).
- specter2 (allenai/specter2_base + adapters): trained on scientific papers
  and their citations, 768 dims. Papers and queries use different adapters
  (asymmetric): `specter2` (proximity) for papers, `specter2_adhoc_query` for
  queries. Paper text = title [SEP] abstract; embedding = the [CLS] vector.
  Adapters are loaded with AdapterHub's `adapters` library, which pins
  transformers to a narrow range: that is why the embed group stays on
  transformers 4.x and sentence-transformers 5.x.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

from papergraph.ingest.text import paper_text

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True)
class EmbeddingModel:
    name: str
    table: str  # where ingest.embed stores paper vectors
    dim: int
    query_instruction: str = ""  # prefix the model expects on retrieval queries


MODELS = {
    "bge-small": EmbeddingModel(
        name="bge-small",
        table="paper_embeddings_bge_small",
        dim=384,
        query_instruction="Represent this sentence for searching relevant passages: ",
    ),
    "specter2": EmbeddingModel(name="specter2", table="paper_embeddings_specter2", dim=768),
}


def get_model(name: str) -> EmbeddingModel:
    try:
        return MODELS[name]
    except KeyError:
        raise ValueError(f"unknown embedding model {name!r}; known: {sorted(MODELS)}") from None


class Encoder(Protocol):
    def encode_papers(
        self, papers: "list[tuple[str, str]]", batch_size: int
    ) -> "NDArray[np.float32]": ...

    def encode_query(self, text: str) -> list[float]: ...


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=2)
def get_encoder(name: str) -> Encoder:
    """The (cached) encoder for a model name. Vectors are L2-normalized."""
    if name == "bge-small":
        return BgeEncoder()
    if name == "specter2":
        return Specter2Encoder()
    raise ValueError(f"unknown embedding model {name!r}")


class BgeEncoder:
    HF_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.HF_NAME, device=device())

    def encode_papers(
        self, papers: list[tuple[str, str]], batch_size: int
    ) -> "NDArray[np.float32]":
        texts = [paper_text(title, abstract) for title, abstract in papers]
        vectors: NDArray[np.float32] = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True
        )
        return vectors

    def encode_query(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vec]


# --- SPECTER2 -----------------------------------------------------------------


class Specter2Encoder:
    BASE = "allenai/specter2_base"
    PAPER_ADAPTER = "allenai/specter2"
    QUERY_ADAPTER = "allenai/specter2_adhoc_query"

    def __init__(self) -> None:
        from transformers import AutoTokenizer

        # Typed as Any: transformers 4.x leaves from_pretrained unannotated, and
        # CI (no embed group) has no transformers to check against at all.
        auto_tokenizer: Any = AutoTokenizer
        self.device = device()
        self.tokenizer = auto_tokenizer.from_pretrained(self.BASE)
        self.paper_model = self._load(self.PAPER_ADAPTER)
        self.query_model = self._load(self.QUERY_ADAPTER)

    def _load(self, adapter_repo: str) -> Any:
        from adapters import AutoAdapterModel

        model = AutoAdapterModel.from_pretrained(self.BASE)
        name = model.load_adapter(adapter_repo, source="hf", set_active=True)
        model.set_active_adapters(name)  # explicit, so a library default can't leave it off
        return model.to(self.device).eval()

    def _encode(self, model: Any, texts: list[str], batch_size: int) -> "NDArray[np.float32]":
        import numpy as np
        import torch

        # Sort by length so each batch pads to similar lengths (much less
        # wasted compute), then restore the input order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            idx = order[start : start + batch_size]
            batch = self.tokenizer(
                [texts[i] for i in idx],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(self.device)
            with torch.no_grad():
                cls = model(**batch).last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, dim=-1)
            out[idx] = cls.float().cpu().numpy()
        return out

    def encode_papers(
        self, papers: list[tuple[str, str]], batch_size: int
    ) -> "NDArray[np.float32]":
        sep = self.tokenizer.sep_token
        texts = [title.strip() + sep + abstract.strip() for title, abstract in papers]
        return self._encode(self.paper_model, texts, batch_size)

    def encode_query(self, text: str) -> list[float]:
        return [float(x) for x in self._encode(self.query_model, [text.strip()], 1)[0]]
