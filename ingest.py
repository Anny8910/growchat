"""Ingestion entry point.

The default run is the full pipeline: fetch/cache -> extract -> chunk -> embed
-> store. `--chunk-only` stops after the Phase 2 artifact so extraction work can
be reviewed without loading the model.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

import config
from rag.chunker import build_chunks, write_chunk_file
from rag.extract import extract_all
from rag.loader import load_all, snapshot_date


def run_chunk_phase(force: bool) -> tuple[int, list]:
    print(f"Loading {len(config.SOURCES)} source pages")
    pages = load_all(force=force)
    for page in pages:
        state = "cache" if page.from_cache else "fetch"
        match = "ok" if page.detected_scheme_code == page.expected_scheme_code else "MISMATCH"
        print(
            f"  [{state:5s}] {page.slug:<44s} code={page.detected_scheme_code} "
            f"({match}) {len(page.text):>6d} chars"
        )

    snapshot = snapshot_date(pages)
    kept, dropped = extract_all(pages)
    print(f"\nExtracted {len(kept)} fact records (snapshot {snapshot})")
    if dropped:
        print(f"  dropped {len(dropped)}:")
        for record in dropped:
            print(f"    {record.scheme_name} / {record.field}: {'; '.join(record.dropped)}")

    chunks = build_chunks(kept, snapshot)
    path = write_chunk_file(chunks, config.SOURCES)

    per_scheme = Counter(chunk.scheme_code for chunk in chunks)
    per_section = Counter(chunk.section for chunk in chunks)
    print(f"\nBuilt {len(chunks)} chunks")
    print("  per scheme :", dict(per_scheme))
    print("  per section:", dict(per_section))
    print(
        f"  size       : max {max(c.word_count for c in chunks)} words, "
        f"max {max(c.char_count for c in chunks)} chars (cap {config.CHUNK_WORDS} words)"
    )
    print(f"\nChunks written to {path}")
    return 0, chunks


def run_store_phase(chunks: list) -> int:
    from rag.embedder import embed_texts, get_model
    from rag.store import (
        StoreError, embedding_preview, stored_dimensions, upsert_chunks,
    )

    print(f"\nLoading {config.EMBED_MODEL} (CPU, first run downloads ~90MB)")
    started = time.monotonic()
    get_model()
    print(f"  model ready in {time.monotonic() - started:.1f}s")

    started = time.monotonic()
    try:
        vectors = embed_texts([chunk.text for chunk in chunks])
    except Exception as exc:
        print(f"Embedding failed: {exc}", file=sys.stderr)
        return 1
    print(f"  embedded {len(vectors)} chunks in {time.monotonic() - started:.1f}s")

    try:
        result = upsert_chunks(chunks, vectors)
    except StoreError as exc:
        print(f"\nStore error:\n{exc}", file=sys.stderr)
        return 1

    print(f"\nStore: {result.added} added, {result.unchanged} unchanged")
    print(f"  collection : {config.CHROMA_COLLECTION} in {config.CHROMA_DIR}")
    print(f"  total docs : {result.store_total}")
    print(f"  dimensions : {sorted(stored_dimensions())} (expected {config.EMBED_DIM})")
    if result.added == 0:
        print("  corpus is already up to date — nothing written")

    try:
        preview = config.EMBEDDINGS_PREVIEW
        preview.parent.mkdir(parents=True, exist_ok=True)
        preview.write_text(
            embedding_preview(limit=5, dims=10), encoding="utf-8"
        )
        print(f"\nEmbeddings preview written to {preview}")
    except Exception as exc:
        print(f"\nWarning: could not write embeddings preview: {exc}", file=sys.stderr)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest the HDFC source corpus.")
    parser.add_argument("--force", action="store_true", help="refetch, ignoring the raw cache")
    parser.add_argument(
        "--chunk-only",
        action="store_true",
        help="stop after writing data/chunks/chunks.txt, skip the model",
    )
    args = parser.parse_args()

    status, chunks = run_chunk_phase(args.force)
    if status != 0:
        return status
    if args.chunk_only:
        print("\n--chunk-only: skipping embed and store")
        return 0
    return run_store_phase(chunks)


if __name__ == "__main__":
    raise SystemExit(main())
