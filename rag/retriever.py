"""Phase 5 retrieval: question -> ranked chunks from the Phase 3 store.

Two constraints from `docs/implementation.md` §Phase 3 constrain this module:

1. **Same encoder (C7).** The question is embedded with `rag.embedder`'s
   process-wide MiniLM instance — the exact same weights and normalisation that
   built the store — and the store refuses incompatible collections. Mixing
   encoders would make cosine distances meaningless.

2. **A lexical scheme filter is required before ranking, and top-1 is
   forbidden.** Pure cosine misranks across schemes ("What is the exit load on
   HDFC Small Cap Fund?" put the *Large Cap* exit-load chunk first), and the
   required fact can sit at rank 5 ("What risk category does HDFC Balanced
   Advantage Fund fall under?"). `retrieve` therefore detects the named scheme,
   restricts Chroma to that scheme's chunks via a metadata `where` filter, and
   returns `TOP_K` chunks with a distance threshold so a question with no good
   match is honestly declared out of scope rather than answered by guesswork
   (F10).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import config
from . import embedder, store

TRIM_PATTERN = re.compile(r"\s+")
_OUT_OF_CORPUS_RE = re.compile(
    r"\b(?:parag|paraag|sbi|axis|icici|nippon|tata|kotak|quant|motilal|"
    r"canara|mid[-\s]?cap|midcap|nifty\s*50|index\s+fund)\b",
    re.I,
)


@dataclass
class SchemeDetection:
    scheme_code: str | None
    out_of_corpus: bool = False
    category: str | None = None


def detect_scheme(question: str) -> SchemeDetection:
    """Map a natural-language question to an allowlisted scheme, if any.

    Returns `out_of_corpus=True` when the question names a fund that is
    definitely not one of the five, e.g. "HDFC Mid-Cap Fund", "Parag Parag
    Flexi Cap", or "HDFC Nifty 50 Index Fund". Those must never retrieve a
    lower-ranked sibling scheme's chunk and answer as if it were the requested
    fund (F10) — "not in my sources" with no number is the only correct answer.

    No detection (no category term, no foreign fund) leaves `scheme_code` None;
    the retriever then searches across all chunks and the distance threshold
    decides scope.
    """
    if _OUT_OF_CORPUS_RE.search(question):
        return SchemeDetection(scheme_code=None, out_of_corpus=True)

    for source in config.SOURCES:
        if _matches_category(question, source["category"]):
            return SchemeDetection(
                scheme_code=source["scheme_code"], category=source["category"]
            )
    return SchemeDetection(scheme_code=None)


def _matches_category(question: str, category: str) -> bool:
    lowered = question.lower()
    canonical = TRIM_PATTERN.sub(" ", category).strip().lower()
    # "tax saver" is the ELSS page's category alias; match it to ELSS.
    if canonical == "elss":
        return bool(re.search(r"\belss\b|\btax\s?saver\b", lowered))
    if canonical == "flexi cap":
        return bool(re.search(r"\b(?:equity[^(]*\()?(?:flexi[-\s]?cap|flexicap)\b", lowered))
    return bool(re.search(rf"\b{re.escape(canonical)}\b", lowered))


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict[str, object]
    distance: float

    @property
    def scheme_code(self) -> str:
        return str(self.metadata["scheme_code"])

    @property
    def scheme_name(self) -> str:
        return str(self.metadata["scheme_name"])

    @property
    def section(self) -> str:
        return str(self.metadata["section"])

    @property
    def fields(self) -> list[str]:
        return str(self.metadata["fields"]).split(",") or []

    @property
    def field_names(self) -> str:
        return ", ".join(self.fields)

    @property
    def source_url(self) -> str:
        return str(self.metadata["source_url"])

    @property
    def retrieved_at(self) -> str:
        return str(self.metadata["retrieved_at"])


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    scheme_code: str | None = None
    out_of_corpus: bool = False
    best_distance: float | None = None

    @property
    def in_scope(self) -> bool:
        """There is a retrieved chunk close enough to answer from."""
        return (
            not self.out_of_corpus
            and self.best_distance is not None
            and self.best_distance <= config.DISTANCE_THRESHOLD
        )

    @property
    def best_chunk(self) -> RetrievedChunk | None:
        return self.chunks[0] if self.chunks else None


def retrieve(question: str, k: int | None = None) -> RetrievalResult:
    """Rank the store's chunks against `question` under the scheme constraint."""
    top_k = k or config.TOP_K
    detection = detect_scheme(question)

    if detection.out_of_corpus:
        return RetrievalResult(chunks=[], out_of_corpus=True)

    vector = embedder.embed_texts([question])[0]

    if detection.scheme_code:
        raw = store.query(
            vector, n_results=top_k, where={"scheme_code": detection.scheme_code}
        )
    else:
        raw = store.query(vector, n_results=top_k)

    chunks = [
        RetrievedChunk(
            id=row["id"], text=row["text"], metadata=row["metadata"], distance=row["distance"]
        )
        for row in raw
    ]
    best = chunks[0].distance if chunks else None
    return RetrievalResult(
        chunks=chunks,
        scheme_code=detection.scheme_code,
        best_distance=best,
    )