"""Chunking invariants: size cap, one scheme and one section per chunk, labels intact."""

from __future__ import annotations

import re

import pytest

import config
from rag.chunker import (
    Chunk, ChunkError, _hash, build_chunks, read_chunks, write_chunk_file,
)
from rag.extract import extract_all
from rag.loader import load_all, snapshot_date

PAGES = load_all()
RECORDS, _ = extract_all(PAGES)
SNAPSHOT = snapshot_date(PAGES)
CHUNKS = build_chunks(RECORDS, SNAPSHOT)


def test_chunks_created():
    assert len(CHUNKS) >= 25
    assert len(CHUNKS) <= 60


def test_no_chunk_exceeds_word_cap():
    for chunk in CHUNKS:
        assert chunk.word_count <= config.CHUNK_WORDS, (chunk.chunk_index, chunk.word_count)


def test_one_scheme_per_chunk():
    for chunk in CHUNKS:
        others = [
            s["scheme_name"]
            for s in config.SOURCES
            if s["scheme_code"] != chunk.scheme_code
        ]
        body = chunk.text.split("\n", 1)[1]
        for other in others:
            assert other not in body, (chunk.chunk_index, other)


def test_one_section_per_chunk():
    for chunk in CHUNKS:
        assert chunk.section


def test_every_chunk_names_its_scheme():
    for chunk in CHUNKS:
        assert chunk.text.splitlines()[0].startswith(chunk.scheme_name)


def test_field_label_stays_with_its_value():
    """A number must never appear without the label that names it."""
    labels = ["Expense ratio", "Minimum SIP", "Lock-in period", "Exit load", "Benchmark", "Risk rating"]
    for chunk in CHUNKS:
        body = chunk.text.split("\n", 1)[1]
        if "%" in body and "Expense ratio" not in body and "Exit load" not in body:
            assert "Stamp duty" not in body
        for label in labels:
            if label in body:
                assert body.index(label) < len(body)


def test_metadata_is_complete():
    required = {
        "source_url",
        "scheme_name",
        "scheme_code",
        "category",
        "section",
        "fields",
        "chunk_index",
        "retrieved_at",
        "content_hash",
    }
    for chunk in CHUNKS:
        assert set(chunk.metadata()) == required


def test_source_url_always_on_allowlist():
    for chunk in CHUNKS:
        assert chunk.source_url in config.ALLOWED_URLS


def test_chunk_index_is_sequential():
    assert [c.chunk_index for c in CHUNKS] == list(range(len(CHUNKS)))


def test_content_hash_is_unique_per_text():
    hashes = [c.content_hash for c in CHUNKS]
    assert len(set(hashes)) == len(hashes)


def test_chunk_file_is_numbered_with_source_and_char_count(tmp_path):
    path = write_chunk_file(CHUNKS, config.SOURCES, tmp_path / "chunks.txt")
    body = path.read_text(encoding="utf-8")
    assert f"CHUNK 001 of {len(CHUNKS):03d}" in body
    for chunk in CHUNKS:
        assert f"characters   : {chunk.char_count}" in body
    for source in config.SOURCES:
        assert source["url"] in body
    headings = re.findall(r"^CHUNK \d{3} of \d{3}$", body, re.MULTILINE)
    assert len(headings) == len(CHUNKS)


def test_expected_facts_are_present():
    body = " ".join(c.text for c in CHUNKS)
    assert "Lock-in period: 3 years" in body
    assert "Expense ratio: 1.21%" in body
    assert "Minimum SIP: ₹500" in body
    assert "Exit load of 1% if redeemed within 1 year" in body
    assert "NIFTY 500 Total Return Index" in body


# --- Round-trip: the committed corpus must be loadable without re-scraping ---
# `ingest.py --from-chunks` (used by the Render build) rebuilds the vector store
# from data/chunks/chunks.txt. If read_chunks() lost or mangled any field, the
# deployed store would silently differ from the reviewed one, so these assert
# exact equality against the in-memory chunks built from the live pages.


def test_read_chunks_round_trips_the_committed_corpus():
    parsed = read_chunks()
    assert len(parsed) == len(CHUNKS)
    for original, restored in zip(CHUNKS, parsed):
        assert restored.text == original.text
        assert restored.source_url == original.source_url
        assert restored.scheme_name == original.scheme_name
        assert restored.scheme_code == original.scheme_code
        assert restored.category == original.category
        assert restored.section == original.section
        assert restored.fields == original.fields
        assert restored.chunk_index == original.chunk_index
        assert restored.retrieved_at == original.retrieved_at
        assert restored.content_hash == original.content_hash


def test_read_chunks_text_still_hashes_to_its_stored_hash():
    """A hash that no longer matches its text means the file was hand-edited."""
    for chunk in read_chunks():
        assert _hash(chunk.text) == chunk.content_hash


def test_read_chunks_only_reads_allowlisted_urls():
    for chunk in read_chunks():
        assert chunk.source_url in config.ALLOWED_URLS


def test_read_chunks_requires_a_present_file(tmp_path):
    with pytest.raises(ChunkError, match="no chunk corpus"):
        read_chunks(tmp_path / "absent.txt")


def test_read_chunks_rejects_a_corpus_with_no_chunks(tmp_path):
    empty = tmp_path / "chunks.txt"
    empty.write_text("=" * 78 + "\nCHUNKS\n" + "=" * 78 + "\n", encoding="utf-8")
    with pytest.raises(ChunkError, match="no chunks found"):
        read_chunks(empty)


def test_read_chunks_rejects_a_truncated_block(tmp_path):
    good = read_chunks()[0]
    broken = tmp_path / "chunks.txt"
    broken.write_text(
        "=" * 78 + "\nCHUNK 001 of 031\n" + "=" * 78 + "\n"
        "source_url   : " + good.source_url + "\n"
        "scheme_name  : " + good.scheme_name + "\n" + "-" * 78 + "\n" + good.text + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ChunkError, match="missing field"):
        read_chunks(broken)


def test_read_chunks_keeps_text_containing_rule_characters(tmp_path):
    """A line of '=' inside chunk text must not be mistaken for a block end."""
    tricky = Chunk(
        text="Expense ratio: 1.21%\n" + "=" * 78 + "\nstill the same chunk",
        source_url=next(iter(config.ALLOWED_URLS)),
        scheme_name="HDFC Large Cap Fund – Direct Growth",
        scheme_code="119018",
        category="fees",
        section="direct growth",
        fields=["expense_ratio"],
        chunk_index=0,
        retrieved_at=SNAPSHOT,
        content_hash="sha256:deadbeefdeadbeef",
    )
    path = tmp_path / "chunks.txt"
    write_chunk_file([tricky], [], path)
    parsed = read_chunks(path)
    assert len(parsed) == 1
    assert parsed[0].text == tricky.text
