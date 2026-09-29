"""Ingestion entry point.

The default run is the full pipeline: fetch/cache -> extract -> chunk -> embed
-> store. `--chunk-only` stops after the Phase 2 artifact so extraction work can
be reviewed without loading the model.
"""

from __future__ import annotations

import argparse
import fcntl
import sys
import time
from collections import Counter
from pathlib import Path

import config
from rag.chunker import ChunkError, build_chunks, read_chunks, write_chunk_file
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


def ensure_store(wait_seconds: int = 300) -> tuple[bool, str]:
    """Build the vector store from the committed corpus if it isn't there yet.

    Streamlit Community Cloud has no build step, so a fresh clone reaches the
    app with an empty data/chroma/ and every visitor would otherwise be told to
    run `python ingest.py` on a machine they don't control. Building on first
    use makes the deploy self-sufficient.

    The build is guarded by an exclusive file lock and re-checked *after* the
    lock is taken, so N browser tabs opening at once produce one build rather
    than N racing writers into the same SQLite file.

    Returns (ok, message). Never raises: the caller decides how to surface a
    failure, and a broken build should read as one sentence, not a traceback.
    """
    if config.chroma_store_exists():
        return True, "store already present"

    lock_path = config.ROOT / "data" / ".store-build.lock"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("w")
    except OSError as exc:
        return False, f"could not create the build lock at {lock_path}: {exc}"

    deadline = time.monotonic() + wait_seconds
    with handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                # Another session is mid-build. Wait for it rather than racing.
                if time.monotonic() >= deadline:
                    return False, (
                        f"timed out after {wait_seconds}s waiting for another "
                        f"session to finish building the vector store"
                    )
                time.sleep(2)

        try:
            # Re-check under the lock: the session that held it may have
            # completed the build while we were waiting.
            if config.chroma_store_exists():
                return True, "store built by another session"

            print(f"No vector store at {config.CHROMA_DIR}; building it now.")
            try:
                chunks = read_chunks()
            except ChunkError as exc:
                return False, (
                    f"cannot build the vector store: {exc}. The committed "
                    f"corpus {config.CHUNKS_FILE} is missing or malformed."
                )
            print(f"Read {len(chunks)} chunks from {config.CHUNKS_FILE} (no fetch)")

            status = run_store_phase(chunks)
            if status != 0:
                return False, (
                    "vector store build failed; see the build log above for "
                    "the underlying error"
                )
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return False, f"vector store build failed: {exc}"
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

    if not config.chroma_store_exists():
        return False, f"vector store build did not produce {config.CHROMA_DIR}"
    return True, "store built"


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest the HDFC source corpus.")
    parser.add_argument("--force", action="store_true", help="refetch, ignoring the raw cache")
    parser.add_argument(
        "--chunk-only",
        action="store_true",
        help="stop after writing data/chunks/chunks.txt, skip the model",
    )
    parser.add_argument(
        "--from-chunks",
        action="store_true",
        help=(
            "skip fetch/extract and build the store from the committed "
            "data/chunks/chunks.txt. Used by the Render build so a deploy "
            "never re-scrapes the source pages."
        ),
    )
    args = parser.parse_args()

    if args.from_chunks:
        from rag.chunker import ChunkError, read_chunks

        try:
            chunks = read_chunks()
        except ChunkError as exc:
            print(f"\nChunk corpus error:\n  {exc}", file=sys.stderr)
            return 1
        print(f"Read {len(chunks)} chunks from {config.CHUNKS_FILE} (no fetch)")
        per_scheme = Counter(chunk.scheme_code for chunk in chunks)
        print("  per scheme :", dict(per_scheme))
        return run_store_phase(chunks)

    status, chunks = run_chunk_phase(args.force)
    if status != 0:
        return status
    if args.chunk_only:
        print("\n--chunk-only: skipping embed and store")
        return 0
    return run_store_phase(chunks)


if __name__ == "__main__":
    raise SystemExit(main())
