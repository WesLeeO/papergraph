"""Embedding models, shared by offline paper embedding (ingest.embed) and
online query embedding (retrieval.dense). Papers and queries must go through
the same code, or their vectors silently stop being comparable.

Models are loaded lazily: importing this module needs no torch.

- bge-small (BAAI/bge-small-en-v1.5): general-purpose, 384 dims, via
  sentence-transformers. Paper text = "title\\nabstract" (ingest.text).
- Other sentence-transformers models (bge-base, gte-base, ...): candidates for
  the M4b embedder sweep. Same paper text; each model gets the query and
  document prefixes it was trained with (an empty prefix where it has none).
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
    hf_name: str = ""  # Hugging Face repo, for sentence-transformers models
    document_prefix: str = ""  # prefix the model expects on documents (papers)
    # Some repos ship their own model code (gte, nomic, arctic). Running it is
    # trusting that repo, so it is opt-in per model, never global.
    trust_remote_code: bool = False
    # Tokens kept per paper. 512 fits nearly every title + abstract and keeps
    # the comparison fair across models with longer limits (gte, Qwen3: 8k+).
    max_seq_length: int = 512
    batch_size: int = 32  # papers per forward pass in ingest.embed


BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# Qwen3-Embedding takes a task description on queries, in this exact format;
# documents get none. The task wording is ours, written once and not tuned.
QWEN3_INSTRUCTION = (
    "Instruct: Given a research question, retrieve the titles and abstracts of "
    "papers that answer it\nQuery:"
)

MODELS = {
    "bge-small": EmbeddingModel(
        name="bge-small",
        table="paper_embeddings_bge_small",
        dim=384,
        query_instruction=BGE_INSTRUCTION,
        hf_name="BAAI/bge-small-en-v1.5",
        batch_size=128,
    ),
    "specter2": EmbeddingModel(name="specter2", table="paper_embeddings_specter2", dim=768),
    # --- M4b sweep candidates -------------------------------------------------
    "bge-base": EmbeddingModel(
        name="bge-base",
        table="paper_embeddings_bge_base",
        dim=768,
        query_instruction=BGE_INSTRUCTION,
        hf_name="BAAI/bge-base-en-v1.5",
    ),
    "bge-large": EmbeddingModel(
        name="bge-large",
        table="paper_embeddings_bge_large",
        dim=1024,
        query_instruction=BGE_INSTRUCTION,
        hf_name="BAAI/bge-large-en-v1.5",
        batch_size=16,
    ),
    "gte-base": EmbeddingModel(
        name="gte-base",
        table="paper_embeddings_gte_base",
        dim=768,
        hf_name="Alibaba-NLP/gte-base-en-v1.5",
        trust_remote_code=True,
    ),
    "arctic-m": EmbeddingModel(
        name="arctic-m",
        table="paper_embeddings_arctic_m",
        dim=768,
        query_instruction="query: ",
        hf_name="Snowflake/snowflake-arctic-embed-m-v2.0",
        trust_remote_code=True,
    ),
    "nomic": EmbeddingModel(
        name="nomic",
        table="paper_embeddings_nomic",
        dim=768,
        query_instruction="search_query: ",
        document_prefix="search_document: ",
        hf_name="nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
    ),
    "qwen3-0.6b": EmbeddingModel(
        name="qwen3-0.6b",
        table="paper_embeddings_qwen3_0_6b",
        dim=1024,
        query_instruction=QWEN3_INSTRUCTION,
        hf_name="Qwen/Qwen3-Embedding-0.6B",
        batch_size=16,
    ),
}


# The model the product searches with (M4b): `make embed` and `make setup`
# embed the corpus with it, and search.py queries its table.
PRODUCT_MODEL = "qwen3-0.6b"


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
    """Best available accelerator: NVIDIA GPU (cuda), Apple GPU (mps), else CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=2)
def get_encoder(name: str) -> Encoder:
    """The (cached) encoder for a model name. Vectors are L2-normalized."""
    if name == "specter2":
        return Specter2Encoder()
    return SentenceTransformerEncoder(get_model(name))


class SentenceTransformerEncoder:
    """Any model sentence-transformers can load. Pooling (CLS, mean or last
    token) comes from the model's own config, so it matches its training."""

    def __init__(self, model: EmbeddingModel) -> None:
        from sentence_transformers import SentenceTransformer

        self.document_prefix = model.document_prefix
        self.model = SentenceTransformer(
            model.hf_name, device=device(), trust_remote_code=model.trust_remote_code
        )
        self.model.max_seq_length = model.max_seq_length

    def encode_papers(
        self, papers: list[tuple[str, str]], batch_size: int
    ) -> "NDArray[np.float32]":
        texts = [self.document_prefix + paper_text(title, abstract) for title, abstract in papers]
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
