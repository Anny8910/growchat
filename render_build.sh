#!/usr/bin/env bash
# Render build step. Run by render.yaml's buildCommand.
#
# Two jobs:
#   1. install deps without the multi-GB CUDA wheel set
#   2. build the vector store, which is git-ignored and therefore absent in
#      a fresh checkout
#
# Idempotent: re-running is safe, and the store write is a no-op when the
# content hashes are already present.

set -euo pipefail

echo "==> Python: $(python --version 2>&1)"

# --- 1. CPU-only torch -------------------------------------------------------
# The default torch wheel on PyPI depends on nvidia-cudnn, nvidia-nccl,
# nvidia-cublas, triton and cuda-toolkit — roughly 3GB of wheels that a CPU
# Render instance can neither use nor finish downloading inside the build
# timeout. Install from the CPU index first so pip is satisfied and
# requirements.txt's `torch==2.14.0` resolves to the already-present build.
#
# Non-fatal on failure: a CUDA torch still works, it is only slower to install.
if pip install --quiet --index-url https://download.pytorch.org/whl/cpu "torch==$(python - <<'PY'
import re, pathlib
line = next(l for l in pathlib.Path("requirements.txt").read_text().splitlines()
            if l.startswith("torch=="))
print(line.split("==", 1)[1].strip())
PY
)"; then
  echo "==> torch: CPU wheel installed"
else
  echo "==> WARNING: CPU torch index unavailable, falling back to default wheel" >&2
fi

echo "==> Installing requirements"
pip install -r requirements.txt

# --- 2. Vector store ---------------------------------------------------------
# Built from the committed data/chunks/chunks.txt, NOT by re-scraping Groww:
# the deployed facts stay identical to the reviewed corpus, and a transient
# fetch block can't fail the deploy.
echo "==> Building vector store from committed corpus"
python ingest.py --from-chunks

# Prove the store is readable before the service is allowed to start, so a
# broken build fails here rather than as a stack trace in the UI.
python - <<'PY'
import config
from rag.retriever import retrieve

if not config.chroma_store_exists():
    raise SystemExit(f"FAIL: no vector store at {config.CHROMA_DIR}")
print(f"==> Store OK: {config.CHROMA_DIR}")

result = retrieve("What is the expense ratio of the HDFC Large Cap Fund?")
if not result.chunks:
    raise SystemExit("FAIL: retrieval returned nothing from a freshly built store")
print(f"==> Retrieval OK: {len(result.chunks)} chunk(s), "
      f"best distance {result.best_distance:.3f}, in_scope={result.in_scope}")
PY

echo "==> Build complete"
