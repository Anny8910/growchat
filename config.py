"""Single source of truth for every tunable, the source allowlist, and paths.

Docs: docs/architecture.md (config), docs/CHUNKING.md (chunk parameters),
docs/PRD.md (the 5 schemes). Importing this module never requires an API key.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")


class MissingConfigError(RuntimeError):
    """Raised when a required setting is absent, with instructions attached."""


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise MissingConfigError(
            f"{name} must be a whole number, got {raw!r}."
        ) from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise MissingConfigError(
            f"{name} must be a number, got {raw!r}."
        ) from exc


SOURCES: list[dict[str, str]] = [
    {
        "slug": "hdfc-large-cap-fund-direct-growth",
        "scheme_name": "HDFC Large Cap Fund – Direct Growth",
        "category": "Large Cap",
        "scheme_code": "119018",
        "url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    },
    {
        "slug": "hdfc-equity-fund-direct-growth",
        "scheme_name": "HDFC Equity (Flexi Cap) Fund – Direct Growth",
        "category": "Flexi Cap",
        "scheme_code": "118955",
        "url": "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
    },
    {
        "slug": "hdfc-elss-tax-saver-fund-direct-plan-growth",
        "scheme_name": "HDFC ELSS Tax Saver Fund – Direct Plan – Growth",
        "category": "ELSS",
        "scheme_code": "119060",
        "url": "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    },
    {
        "slug": "hdfc-small-cap-fund-direct-growth",
        "scheme_name": "HDFC Small Cap Fund – Direct Growth",
        "category": "Small Cap",
        "scheme_code": "130503",
        "url": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    },
    {
        "slug": "hdfc-balanced-advantage-fund-direct-growth",
        "scheme_name": "HDFC Balanced Advantage Fund – Direct Growth",
        "category": "Balanced Advantage",
        "scheme_code": "118968",
        "url": "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
    },
]

ALLOWED_URLS: frozenset[str] = frozenset(s["url"] for s in SOURCES)
ALLOWED_SCHEME_CODES: frozenset[str] = frozenset(s["scheme_code"] for s in SOURCES)
EXPECTED_SCHEME_CODES: dict[str, str] = {
    s["slug"]: s["scheme_code"] for s in SOURCES
}

EXAMPLE_QUESTIONS: list[str] = [
    "What is the expense ratio of the HDFC Large Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
    "What is the minimum SIP amount for HDFC Balanced Advantage Fund?",
]

DISCLAIMER = "Facts-only. No investment advice."

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
HTTP_TIMEOUT = 30

EMBED_MODEL = _env_str("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = _env_int("EMBED_DIM", 384)

CHUNK_WORDS = _env_int("CHUNK_WORDS", 150)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 25)

TOP_K = _env_int("TOP_K", 5)

# Conversation memory: how many recent user/assistant messages (a question and
# its answer count as two) are kept in RAM to resolve follow-up questions. This
# is deliberately in-process only and never persisted (C12).
CONVERSATION_MESSAGES = _env_int("CONVERSATION_MESSAGES", 10)
DISTANCE_THRESHOLD = _env_float("DISTANCE_THRESHOLD", 0.45)

GROQ_MODEL = _env_str("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = _env_float("GROQ_TEMPERATURE", 0.0)
# 1024, not the ~250 an answer actually needs, because GROQ_MODEL is a reasoning
# model and its reasoning tokens are counted against this budget. At 250 the
# model sometimes spent the whole allowance thinking and returned truncated text
# (or nothing), which validate_output caught and answered as "I don't know" —
# measured 3 failures in 8 runs at 250 versus 0 in 8 at 1024. Raising this does
# not loosen the answer contract: the 3-sentence cap and the single-citation
# rule are enforced by validate_output, independently of the token budget.
GROQ_MAX_TOKENS = _env_int("GROQ_MAX_TOKENS", 1024)
GROQ_API_KEY = _env_str("GROQ_API_KEY", "")

RAW_DIR = ROOT / "data" / "raw"
CHUNKS_DIR = ROOT / "data" / "chunks"
CHUNKS_FILE = CHUNKS_DIR / "chunks.txt"
EMBEDDINGS_PREVIEW = ROOT / "data" / "embeddings_preview.txt"
CHROMA_DIR = ROOT / "data" / "chroma"
CHROMA_COLLECTION = _env_str("CHROMA_COLLECTION", "hdfc_faqs")
STATIC_DIR = ROOT / "static"

ensure_data_dirs: list[Path] = [RAW_DIR, CHROMA_DIR]


def require_groq_key() -> str:
    if not GROQ_API_KEY:
        raise MissingConfigError(
            "GROQ_API_KEY is not set.\n"
            "  1. Get a free key at https://console.groq.com/keys\n"
            "  2. cp .env.example .env\n"
            "  3. Add GROQ_API_KEY=<your key> to .env\n"
            "Phases 1-4 (ingestion, vector store, guardrails) do not need this."
        )
    return GROQ_API_KEY


def chroma_store_exists() -> bool:
    return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())


def print_settings() -> None:
    print("Sources (allowlist — the only URLs that may be cited):")
    for s in SOURCES:
        print(f"  [{s['scheme_code']}] {s['category']:22s} {s['scheme_name']}")
        print(f"      {s['url']}")
    print(f"\nEmbedding model   : {EMBED_MODEL} ({EMBED_DIM}-dim)")
    print(f"Chunking          : {CHUNK_WORDS} words, {CHUNK_OVERLAP} overlap")
    print(f"Retrieval         : top_k={TOP_K}, distance_threshold={DISTANCE_THRESHOLD}")
    print(
        f"Generation        : {GROQ_MODEL}, temperature={GROQ_TEMPERATURE}, "
        f"max_tokens={GROQ_MAX_TOKENS}"
    )
    print(f"Conversation      : last {CONVERSATION_MESSAGES} messages (in memory only)")
    print(f"Groq API key      : {'set' if GROQ_API_KEY else 'not set (fine until Phase 5)'}")
    print(f"Disclaimer        : {DISCLAIMER}")
    print(f"Chroma dir        : {CHROMA_DIR} (exists={chroma_store_exists()})")
    print(f"Chunks file       : {CHUNKS_FILE}")
    print(f"Embeddings preview: {EMBEDDINGS_PREVIEW}")
    print(f"Raw cache dir     : {RAW_DIR}")
