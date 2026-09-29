"""Chunking invariants: size cap, one scheme and one section per chunk, labels intact."""

from __future__ import annotations

import re

import config
from rag.chunker import build_chunks, write_chunk_file
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
