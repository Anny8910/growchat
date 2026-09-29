"""Local MiniLM encoder.

The model is loaded lazily and cached for the life of the process. Loading it
costs ~2s and ~90MB of memory, so `import rag.embedder` must stay cheap — the
retriever and the store both share this one instance.

Why one shared instance matters (C7): cosine distance is only meaningful
between vectors produced by the same encoder. Query and document embeddings
must come from identical weights and identical normalisation. If the store was
built with a different model, distances silently stop being comparable, so
`store.py` records the model name in the collection metadata and refuses to
mix them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import config

_MODEL = None


class EmbeddingError(RuntimeError):
    """Raised when the encoder cannot produce the configured dimensionality."""


def model_cache_dir() -> Path | None:
    """Where huggingface_hub keeps this model's weights, if anywhere."""
    root = os.getenv("HF_HUB_CACHE") or (
        Path.home() / ".cache" / "huggingface" / "hub"
    )
    return Path(root) / ("models--" + config.EMBED_MODEL.replace("/", "--"))


def model_is_cached() -> bool:
    """True when the weights are already on disk.

    Without this, sentence-transformers issues a HEAD request on every load to
    check for a newer revision. That is invisible when the network works and
    expensive when it does not: a blocked run spent 145s in retry backoff for
    a model it already had. Checking the cache directory ourselves turns
    "offline after the first download" from a claim into a fact.
    """
    cache = model_cache_dir()
    if cache is None or not cache.is_dir():
        return False
    return any(cache.rglob("*.safetensors")) or any(cache.rglob("*.bin"))


def get_model():
    """Return the process-wide SentenceTransformer, loading it on first use."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        cached = model_is_cached()
        _MODEL = SentenceTransformer(
            config.EMBED_MODEL,
            device="cpu",
            local_files_only=cached,
        )
        if cached:
            print("  using locally cached model weights (no network)")
    return _MODEL


def model_name() -> str:
    return config.EMBED_MODEL


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed texts to unit-length vectors of config.EMBED_DIM dimensions.

    Normalised so that cosine *distance* reduces to 1 - dot product, which is
    what Chroma's `cosine` space expects and what Phase 5 thresholds are
    calibrated against.
    """
    if not texts:
        return []

    vectors = get_model().encode(
        list(texts),
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    dim = len(vectors[0])
    if dim != config.EMBED_DIM:
        raise EmbeddingError(
            f"{config.EMBED_MODEL} produced {dim}-dim vectors but "
            f"config.EMBED_DIM is {config.EMBED_DIM}. Every existing vector in "
            f"{config.CHROMA_DIR} was built at the old dimension, so cosine "
            f"distances are no longer comparable.\n"
            f"  Fix: set EMBED_DIM={dim} in .env, then delete "
            f"{config.CHROMA_DIR} and re-run `python ingest.py --store`."
        )
    return [vector.tolist() for vector in vectors]
