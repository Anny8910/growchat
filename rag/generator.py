"""Phase 5 generation: retrieved chunks -> grounded, cited Groq answer.

`answer()` is the full query path described in `docs/implementation.md`
§Phase 5: guardrail first, then retrieve, then a deterministic Groq call fed
only the retrieved context, then `validate_output` so a contract violation
becomes a safe fallback rather than bad text on screen.

Every user-facing refusal and the answer contract live in `rag/prompts.py`,
so the CLI, the README, and `docs/sample_qa.md` cannot drift apart.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import config
from . import prompts
from .guardrails import check_question, validate_output
from .retriever import RetrievalResult, retrieve

RETRYABLE_STATUS = (429, 500, 502, 503, 504)
RETRY_ATTEMPTS = 2
RETRY_DELAY = 1.0


class GenerationError(RuntimeError):
    """Raised when the LLM call fails and no fallback can be produced."""


@dataclass
class Answer:
    question: str
    text: str
    citation: str | None
    ok: bool
    kind: str  # "answer" | "refused" | "not_in_sources" | "fallback" | "error"
    retrieved: RetrievalResult
    violations: list[str] = field(default_factory=list)
    guardrail_kind: str = "ok"  # "ok" | "pii" | "advice" | "cross_scheme" | "off_topic"
    rewritten_question: str | None = None  # set when a follow-up was resolved

    @property
    def refused(self) -> bool:
        return self.kind in ("refused", "not_in_sources", "fallback", "error")


def build_user_prompt(question: str, chunks: list) -> str:
    return prompts.build_user_prompt(question, chunks)


def call_groq(system: str, user: str, client=None) -> str:
    """One deterministic Groq call with a single retry on transient failure.

    Returns the raw assistant text. `client` is injectable for tests; anything
    that is not retryable — an auth failure, a bad model id — propagates as a
    `GenerationError` with the API's message attached, so the caller can show a
    clear error instead of a fabricated answer (Phase 5 step 11).

    An injected client skips the key lookup, so tests (and the conversation
    rewriter's fallback path) can exercise the real code with no credentials.
    """
    if client is None:
        api_key = config.require_groq_key()
        from groq import Groq

        client = Groq(api_key=api_key)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    last_error: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=config.GROQ_MODEL,
                messages=messages,
                temperature=config.GROQ_TEMPERATURE,
                max_tokens=config.GROQ_MAX_TOKENS,
            )
            text = response.choices[0].message.content
            return text or ""
        except Exception as exc:
            last_error = exc
            status = getattr(exc, "status_code", None)
            if status not in RETRYABLE_STATUS or attempt == RETRY_ATTEMPTS - 1:
                break
            time.sleep(RETRY_DELAY)

    assert last_error is not None
    status = getattr(last_error, "status_code", None)
    message = f"{type(last_error).__name__}: {last_error}"
    if status is not None:
        message = f"Groq API error {status}: {last_error}"
    raise GenerationError(message) from last_error


def _not_in_sources_answer(question: str, retrieved: RetrievalResult) -> Answer:
    scheme_url = None
    if retrieved.scheme_code:
        for source in config.SOURCES:
            if source["scheme_code"] == retrieved.scheme_code:
                scheme_url = source["url"]
                break
    text = prompts.not_in_sources_message(scheme_url)
    return Answer(
        question=question,
        text=text,
        citation=scheme_url,
        ok=False,
        kind="not_in_sources",
        retrieved=retrieved,
    )


def _citation_for(retrieved: RetrievalResult) -> str | None:
    """The single allowlisted citation for a grounded answer.

    The best chunk's `source_url` is its scheme page, which is by construction
    inside `config.ALLOWED_URLS`, so attaching exactly this URL always passes
    `validate_output`'s allowlist check.
    """
    best = retrieved.best_chunk
    return best.source_url if best else None


def answer(question: str, client=None) -> Answer:
    """Full path: guardrail -> retrieve -> generate -> validate -> fallback."""
    decision = check_question(question)
    if decision.refused:
        text = decision.message
        if decision.link:
            text += f"\n{decision.link}"
        return Answer(
            question=question,
            text=text,
            citation=decision.link,
            ok=False,
            kind="refused",
            retrieved=RetrievalResult(chunks=[]),
            guardrail_kind=decision.kind,
        )

    retrieved = retrieve(question)
    if not retrieved.in_scope or retrieved.out_of_corpus:
        return _not_in_sources_answer(question, retrieved)

    system = prompts.SYSTEM_PROMPT
    user = build_user_prompt(question, retrieved.chunks)

    try:
        raw = call_groq(system, user, client=client)
    except GenerationError as exc:
        fallback = prompts.NO_CONTEXT_REFUSAL + f"\n{prompts.NOT_IN_SOURCES_LINK}"
        return Answer(
            question=question,
            text=fallback,
            citation=prompts.NOT_IN_SOURCES_LINK,
            ok=False,
            kind="error",
            retrieved=retrieved,
            violations=[str(exc)],
        )

    citation = _citation_for(retrieved)
    updated = retrieved.best_chunk.retrieved_at
    assembled = f"{raw.strip()}\n{prompts.last_updated_line(updated)}\n{citation}"

    check = validate_output(assembled)
    if not check.ok:
        fallback = prompts.NO_CONTEXT_REFUSAL + f"\n{prompts.NOT_IN_SOURCES_LINK}"
        return Answer(
            question=question,
            text=fallback,
            citation=prompts.NOT_IN_SOURCES_LINK,
            ok=False,
            kind="fallback",
            retrieved=retrieved,
            violations=check.violations,
        )

    return Answer(
        question=question,
        text=assembled,
        citation=citation,
        ok=check.ok,
        kind="answer",
        retrieved=retrieved,
    )