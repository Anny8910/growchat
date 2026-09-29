"""Vector store tests.

Two groups:
  * contract tests against the real `data/chroma` store (the Phase 3 deliverable)
  * behavioural tests against a throwaway store in a temp dir, which need no
    model download and run in milliseconds

The behavioural tests inject dummy 384-dim vectors rather than real embeddings:
`upsert_chunks` is a store operation, so testing it through the model would
only add a 90MB download and slow the suite down.
"""

from __future__ import annotations

import re

import pytest

import config
from rag.chunker import Chunk, _hash
from rag.store import (
    StoreError, count, embedding_preview, get_client, get_collection,
    read_vectors, reset_client, upsert_chunks,
)

CHUNK_COUNT = 31
DUMMY = [[0.0] * config.EMBED_DIM for _ in range(1)]


def make_chunk(text: str = "HDFC Test Fund — Large Cap\nExpense ratio: 0.50%.", **kw) -> Chunk:
    defaults = dict(
        text=text,
        source_url=config.SOURCES[0]["url"],
        scheme_name=config.SOURCES[0]["scheme_name"],
        scheme_code=config.SOURCES[0]["scheme_code"],
        category=config.SOURCES[0]["category"],
        section="overview",
        fields=["expense_ratio"],
        chunk_index=0,
        retrieved_at="2026-09-29",
        content_hash=_hash(text),
    )
    defaults.update(kw)
    return Chunk(**defaults)


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    """An isolated store so idempotency tests never touch the real corpus."""
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "CHROMA_COLLECTION", "test_collection")
    reset_client()
    yield config.CHROMA_DIR
    reset_client()


# --- contract: the real store ------------------------------------------------

def test_store_exists_and_is_populated():
    assert config.chroma_store_exists(), "run `python ingest.py` first"
    assert count() == CHUNK_COUNT


def test_store_count_matches_chunks_file():
    """F8: the store must never silently drift from the graded artifact."""
    text = config.CHUNKS_FILE.read_text(encoding="utf-8")
    in_file = int(re.search(r"of (\d+)", text).group(1))
    assert count() == in_file == CHUNK_COUNT


def test_every_vector_is_384_dim():
    stored = get_collection().get(include=["embeddings"])
    dims = {len(vec) for vec in stored["embeddings"]}
    assert dims == {config.EMBED_DIM}


def test_collection_records_its_encoder():
    """C7: the encoder identity is what makes cosine distances meaningful."""
    meta = get_collection().metadata
    assert meta["hnsw:space"] == "cosine"
    assert meta["embed_model"] == config.EMBED_MODEL
    assert meta["embed_dim"] == config.EMBED_DIM


def test_every_record_has_all_metadata_fields():
    required = {
        "source_url", "scheme_name", "scheme_code", "category",
        "section", "fields", "chunk_index", "retrieved_at", "content_hash",
    }
    records = get_collection().get(include=["metadatas"])["metadatas"]
    assert len(records) == CHUNK_COUNT
    for meta in records:
        assert not (required - meta.keys()), f"missing {required - meta.keys()}"
        assert all(str(meta[k]).strip() for k in required), meta


def test_stored_citations_are_allowlisted():
    """Compliance: nothing outside the 5 approved URLs may be citable."""
    records = get_collection().get(include=["metadatas"])["metadatas"]
    urls = {m["source_url"] for m in records}
    assert urls <= config.ALLOWED_URLS
    assert {m["scheme_code"] for m in records} <= config.ALLOWED_SCHEME_CODES


def test_store_survives_a_fresh_client_handle():
    reset_client()
    assert count() == CHUNK_COUNT
    assert get_collection().name == config.CHROMA_COLLECTION


# --- the embeddings preview artefact -----------------------------------------

def test_preview_file_was_written():
    assert config.EMBEDDINGS_PREVIEW.exists(), "run `python ingest.py` first"


def test_preview_shows_five_vectors_of_ten_dims():
    text = config.EMBEDDINGS_PREVIEW.read_text(encoding="utf-8")
    assert text.count("chunk_index   :") == 5
    for line in text.splitlines():
        if line.startswith("first 10 dims :"):
            values = [v for v in line.split(":", 1)[1].split(",")]
            assert len(values) == 10
            for value in values:
                assert -1.0 <= float(value) <= 1.0


def test_preview_reports_the_true_stored_count():
    """The header must not drift from the collection it describes."""
    text = config.EMBEDDINGS_PREVIEW.read_text(encoding="utf-8")
    reported = int(re.search(r"vectors stored: (\d+)", text).group(1))
    assert reported == count() == CHUNK_COUNT


def test_preview_vectors_are_normalised_and_complete():
    """Normalisation is what makes cosine distance a dot product (C7)."""
    ids = [
        m["content_hash"]
        for m in get_collection().get(include=["metadatas"])["metadatas"]
    ]
    vectors = read_vectors(sorted(ids)[:5])
    assert len(vectors) == 5
    for vector in vectors.values():
        assert len(vector) == config.EMBED_DIM
        assert sum(c * c for c in vector) ** 0.5 == pytest.approx(1.0, abs=1e-5)


def test_read_vectors_rejects_unknown_ids():
    with pytest.raises(KeyError, match="not in the store"):
        read_vectors(["sha256:does-not-exist"])


def test_preview_handles_an_empty_store(temp_store):
    assert "Store is empty" in embedding_preview()


# --- behaviour: throwaway store ----------------------------------------------

def test_upsert_adds_then_is_idempotent(temp_store):
    chunk = make_chunk()
    first = upsert_chunks([chunk], DUMMY)
    assert (first.added, first.unchanged, first.store_total) == (1, 0, 1)

    second = upsert_chunks([chunk], DUMMY)
    assert (second.added, second.unchanged, second.store_total) == (0, 1, 1)


def test_changed_text_creates_a_new_record(temp_store):
    """A page edit must not be silently skipped as a duplicate."""
    upsert_chunks([make_chunk("HDFC Test Fund — Large Cap\nExpense ratio: 0.50%.")], DUMMY)
    result = upsert_chunks([make_chunk("HDFC Test Fund — Large Cap\nExpense ratio: 0.62%.")], DUMMY)
    assert result.added == 1
    assert result.store_total == 2


def test_mismatched_embedding_count_is_rejected(temp_store):
    with pytest.raises(StoreError, match="one-to-one"):
        upsert_chunks([make_chunk(), make_chunk("other text")], DUMMY)


def test_refuses_to_mix_a_different_encoder(temp_store):
    """The failure this prevents is silent, so it must be loud."""
    # Must go through our own client: Chroma refuses two clients on one path
    # with different settings.
    get_client().get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine", "embed_model": "some-other-model", "embed_dim": 384},
    )
    reset_client()
    with pytest.raises(StoreError, match="different encoder configuration"):
        get_collection()
