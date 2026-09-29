"""Persistent ChromaDB vector store for the approved chunk corpus.

The store is the Phase 3 deliverable and the app's read path. Ingestion is a
separate command from serving: nothing here ever re-embeds.

Idempotency (F8): each chunk's Chroma id is its `content_hash`, so re-running
ingestion with unchanged source text is a no-op. An accidental double-run
cannot double the corpus.

Encoder identity is recorded in the collection metadata and checked on every
open. Chroma has no `modify_collection` in 1.5.9, so a model change cannot be
patched in place — it has to be a deliberate rebuild. We fail loudly instead
of writing new-model vectors into an old-model collection, which is the exact
failure that produces meaningless distances and hours of confused debugging.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from .chunker import Chunk

_CLIENT = None
_COLLECTION = None


class StoreError(RuntimeError):
    """Raised when the on-disk store cannot safely be used as-is."""


@dataclass
class UpsertResult:
    added: int
    unchanged: int
    store_total: int


def reset_client() -> None:
    """Drop cached client handles. Used by tests and after a rebuild."""
    global _CLIENT, _COLLECTION
    _CLIENT = None
    _COLLECTION = None


def _expected_metadata() -> dict[str, object]:
    return {
        "hnsw:space": "cosine",
        "embed_model": config.EMBED_MODEL,
        "embed_dim": int(config.EMBED_DIM),
    }


def get_client():
    global _CLIENT
    if _CLIENT is None:
        import chromadb
        from chromadb.config import Settings

        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _CLIENT = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
    return _CLIENT


def get_collection():
    """Open the collection, creating it if absent, and verify it is compatible."""
    global _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION

    client = get_client()
    wanted = _expected_metadata()
    existing = {c.name: c for c in client.list_collections()}

    if config.CHROMA_COLLECTION in existing:
        collection = client.get_collection(config.CHROMA_COLLECTION)
        actual = collection.metadata or {}
        stale = {
            key: (actual.get(key), value)
            for key, value in wanted.items()
            if key in actual and actual[key] != value
        }
        if stale:
            detail = ", ".join(
                f"{key}: store has {got!r}, config wants {want!r}"
                for key, (got, want) in stale.items()
            )
            raise StoreError(
                f"The store in {config.CHROMA_DIR} was built with a different "
                f"encoder configuration ({detail}). Mixing these would make "
                f"cosine distances meaningless (C7), and Chroma cannot change a "
                f"collection's config in place.\n"
                f"  Fix: delete {config.CHROMA_DIR} and re-run "
                f"`python ingest.py --store`. Chunk text is unchanged, so this "
                f"only costs the embedding step."
            )
        _COLLECTION = collection
        return _COLLECTION

    _COLLECTION = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata=wanted,
    )
    return _COLLECTION


def existing_hashes() -> set[str]:
    return set(get_collection().get(include=[])["ids"])


def upsert_chunks(chunks: list[Chunk], embeddings: list[list[float]]) -> UpsertResult:
    """Write chunks, skipping any whose content_hash is already stored.

    Caller supplies embeddings so this stays a pure store operation and the
    same vectors are used for both the write and the later verification.
    """
    if len(chunks) != len(embeddings):
        raise StoreError(
            f"got {len(chunks)} chunks but {len(embeddings)} embeddings; "
            f"they must correspond one-to-one"
        )

    collection = get_collection()
    known = existing_hashes()
    to_add = [c for c in chunks if c.content_hash not in known]

    if to_add:
        collection.upsert(
            ids=[c.content_hash for c in to_add],
            embeddings=[e for c, e in zip(chunks, embeddings) if c.content_hash not in known],
            documents=[c.text for c in to_add],
            metadatas=[c.metadata() for c in to_add],
        )

    return UpsertResult(
        added=len(to_add),
        unchanged=len(chunks) - len(to_add),
        store_total=collection.count(),
    )


def count() -> int:
    return get_collection().count()


def query(vector: list[float], n_results: int, where: dict | None = None) -> list[dict]:
    """Cosine query, optionally restricted to matching metadata.

    The `where` filter is how the Phase 5 retriever enforces the scheme-name
    constraint documented in `docs/implementation.md` §Phase 3: a question
    about HDFC ELSS can never return a Small Cap chunk, even when one scores
    marginally higher, because the filter is applied before ranking.
    """
    response = get_collection().query(
        query_embeddings=[vector],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = response["ids"][0]
    return [
        {
            "id": ids[i],
            "text": response["documents"][0][i],
            "metadata": response["metadatas"][0][i],
            "distance": response["distances"][0][i],
        }
        for i in range(len(ids))
    ]


def stored_dimensions() -> set[int]:
    """Dimensions actually present in the store, read back from disk."""
    stored = get_collection().get(include=["embeddings"])
    vectors = stored["embeddings"]
    if vectors is None:
        return set()
    return {len(vec) for vec in vectors}


def read_vectors(ids: list[str]) -> dict[str, list[float]]:
    """Fetch stored vectors by id. Raises KeyError for ids not on disk."""
    stored = get_collection().get(ids=ids, include=["embeddings"])
    found = stored["ids"]
    missing = set(ids) - set(found)
    if missing:
        raise KeyError(f"{len(missing)} chunk(s) are not in the store: {sorted(missing)[:3]}")
    return {id_: list(vec) for id_, vec in zip(found, stored["embeddings"])}


def embedding_preview(limit: int = 5, dims: int = 10) -> str:
    """Human-readable dump of the first `limit` stored vectors, `dims` each.

    Read back out of Chroma rather than reused from the in-flight batch, so
    this file is evidence of what is actually on disk and not merely of what
    the encoder returned. Dimensionality is a correctness property of the
    persisted corpus (C7), and a preview built from memory could not catch a
    store that silently truncated or padded its vectors.
    """
    collection = get_collection()
    stored = collection.get(include=["embeddings", "metadatas"])
    ids = stored["ids"]
    vectors = stored["embeddings"]
    metas = stored["metadatas"]

    if ids and vectors is not None:
        order = sorted(range(len(ids)), key=lambda i: metas[i]["chunk_index"])
        rows = list(zip([ids[i] for i in order], vectors, [metas[i] for i in order]))
    else:
        rows = []

    total = len(rows)
    shown = rows[:limit]

    lines = [
        "=" * 78,
        "EMBEDDINGS PREVIEW",
        "=" * 78,
        f"model         : {config.EMBED_MODEL}",
        f"collection    : {config.CHROMA_COLLECTION} ({config.CHROMA_DIR})",
        f"space         : cosine (lower distance = more similar)",
        f"vectors stored: {total}",
        f"showing       : first {len(shown)} vector(s), first {dims} of each dimension",
        "",
        "Each vector below was read back from the persisted Chroma store, not",
        "reused from the in-memory batch, so this file reflects disk state.",
        "Full vectors are 384-dimensional and unit-length (L2 norm ~1.0) because",
        "the encoder normalises; only the leading dimensions are shown here.",
        "",
    ]

    if not shown:
        lines.append("Store is empty - run `python ingest.py` first.")
        return "\n".join(lines) + "\n"

    for id_, vector, meta in shown:
        head = vector[:dims]
        norm = sum(component**2 for component in vector) ** 0.5
        lines.extend(
            [
                "-" * 78,
                f"chunk_index   : {meta['chunk_index']}",
                f"scheme        : {meta['scheme_name']}  ({meta['scheme_code']})",
                f"category      : {meta['category']}",
                f"section       : {meta['section']}",
                f"content_hash  : {id_}",
                f"dimensions    : {len(vector)}  (L2 norm {norm:.6f})",
                f"first {dims:<2d} dims : " + ", ".join(f"{value:+.5f}" for value in head),
                "",
            ]
        )

    lines.append("=" * 78)
    return "\n".join(lines) + "\n"

