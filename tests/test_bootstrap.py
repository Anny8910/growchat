"""First-use bootstrap: building the store on a Cloud deploy, and reading secrets.

Streamlit Community Cloud runs `streamlit run app.py` with no build step, so a
fresh clone has no vector store and its secrets live in `st.secrets` rather than
the process environment. These cover both gaps.
"""

from __future__ import annotations

import pytest

import config
import ingest
from ingest import ensure_store


@pytest.fixture
def no_store(tmp_path, monkeypatch):
    """Point the store and corpus at a tmp dir holding only the chunk file."""
    chroma = tmp_path / "chroma"
    monkeypatch.setattr(config, "CHROMA_DIR", chroma)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(
        config, "CHUNKS_FILE", tmp_path / "data" / "chunks.txt"
    )
    monkeypatch.setattr(
        config, "EMBEDDINGS_PREVIEW", tmp_path / "data" / "preview.txt"
    )
    return tmp_path


def test_ensure_store_is_a_no_op_when_the_store_exists(monkeypatch):
    monkeypatch.setattr(config, "chroma_store_exists", lambda: True)
    assert ensure_store() == (True, "store already present")


def test_ensure_store_builds_from_the_committed_corpus(no_store, monkeypatch):
    """The Cloud path: corpus is committed, store is not. No scraping allowed."""
    no_store.joinpath("data", "chunks.txt").write_text(
        _corpus_for(no_store), encoding="utf-8"
    )
    # Any fetch attempt during a deploy-time build is a bug, not a slow path.
    def explode(*_a, **_k):
        raise AssertionError("ensure_store must not fetch source pages")

    monkeypatch.setattr(ingest, "load_all", explode)
    monkeypatch.setattr(ingest, "run_store_phase", lambda chunks: (
        no_store.joinpath("chroma").mkdir(parents=True, exist_ok=True),
        no_store.joinpath("chroma", "x.bin").write_bytes(b"x"),
        0,
    )[-1])

    ok, detail = ensure_store()
    assert ok, detail
    assert detail == "store built"


def test_ensure_store_reports_a_missing_corpus_instead_of_raising(no_store, monkeypatch):
    monkeypatch.setattr(config, "chroma_store_exists", lambda: False)
    ok, detail = ensure_store()
    assert ok is False
    assert "missing or malformed" in detail


def test_ensure_store_reports_a_failed_build_instead_of_raising(no_store, monkeypatch):
    no_store.joinpath("data", "chunks.txt").write_text(
        _corpus_for(no_store), encoding="utf-8"
    )
    monkeypatch.setattr(config, "chroma_store_exists", lambda: False)
    monkeypatch.setattr(ingest, "run_store_phase", lambda chunks: 1)
    ok, detail = ensure_store()
    assert ok is False
    assert "build failed" in detail


def test_ensure_store_reports_an_exploding_build_instead_of_raising(no_store, monkeypatch):
    """A stack trace in the chat window is the outcome being avoided."""
    no_store.joinpath("data", "chunks.txt").write_text(
        _corpus_for(no_store), encoding="utf-8"
    )
    monkeypatch.setattr(config, "chroma_store_exists", lambda: False)

    def boom(_chunks):
        raise RuntimeError("chromadb exploded")

    monkeypatch.setattr(ingest, "run_store_phase", boom)
    ok, detail = ensure_store()
    assert ok is False
    assert "chromadb exploded" in detail


def _corpus_for(tmp_path) -> str:
    """A minimal two-chunk corpus in the on-disk format."""
    from rag.chunker import Chunk, format_chunk_file

    url = next(iter(config.ALLOWED_URLS))
    chunks = [
        Chunk(
            text="Expense ratio: 1.21%",
            source_url=url,
            scheme_name="HDFC Large Cap Fund – Direct Growth",
            scheme_code="119018",
            category="fees",
            section="direct growth",
            fields=["expense_ratio"],
            chunk_index=i,
            retrieved_at="2026-01-01",
            content_hash=f"sha256:{'0' * 16}",
        )
        for i in range(2)
    ]
    return format_chunk_file(chunks, [])
