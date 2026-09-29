"""Encoder tests that do not need the model weights.

`embed_texts` is deliberately not exercised here: it needs a ~90MB download and
Phase 3 verification already asserts the real store is 384-dim, which is the
property that actually matters. These tests cover the cheap logic that decides
whether we reach for the network at all.
"""

from __future__ import annotations

import pytest

import config
from rag.embedder import EmbeddingError, model_cache_dir, model_is_cached


def test_cache_dir_uses_configured_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert model_cache_dir() == tmp_path / (
        "models--" + config.EMBED_MODEL.replace("/", "--")
    )


def test_model_not_cached_when_directory_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert model_is_cached() is False


def test_model_not_cached_when_only_metadata_present(tmp_path, monkeypatch):
    """A cache dir with no weights must not be mistaken for a usable model."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / ("models--" + config.EMBED_MODEL.replace("/", "--"))
    (repo / "snapshots" / "abc").mkdir(parents=True)
    (repo / "snapshots" / "abc" / "config.json").write_text("{}")
    assert model_is_cached() is False


def test_model_cached_when_weights_present(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / ("models--" + config.EMBED_MODEL.replace("/", "--"))
    snap = repo / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"")
    assert model_is_cached() is True


def test_wrong_dimension_fails_loudly():
    """The dimension guard must explain the fix, not just raise."""
    with pytest.raises(EmbeddingError) as exc:
        from rag import embedder

        original = embedder.get_model

        class Fake:
            def encode(self, texts, **kw):
                return [[0.0] * 7 for _ in texts]

        embedder.get_model = lambda: Fake()
        try:
            embedder.embed_texts(["x"])
        finally:
            embedder.get_model = original

    message = str(exc.value)
    assert "7-dim" in message and str(config.EMBED_DIM) in message
    assert "delete" in message.lower()


def test_embed_empty_list_short_circuits():
    from rag.embedder import embed_texts

    assert embed_texts([]) == []
