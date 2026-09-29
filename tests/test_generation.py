"""Generation tests: prompt assembly, Groq call, retry, refusals, fallback.

The Groq call is tested with an *injected fake client* — real calls consume
free-tier quota and depend on network, so the suite asserts the contract, the
retry, and the fallback deterministically. The one live sanity check is
skipped unless `RUN_LIVE_GROQ=1`.

The fake mirrors the shape used by the real SDK (`client.chat.completions
.create`) so no code path is special-cased for tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

import config
from rag import prompts
from rag.generator import (
    Answer, GenerationError, RETRY_ATTEMPTS, call_groq, answer, _citation_for, build_user_prompt,
)
from rag.guardrails import validate_output
from rag.retriever import retrieve


# --- fake Groq client --------------------------------------------------------

@dataclass
class Choice:
    message: object


@dataclass
class _Content:
    content: str


@dataclass
class _Message:
    content: str


class FakeCompletions:
    def __init__(self, responses, retry_once_first=False):
        self._responses = list(responses)
        self.calls = []
        self._retry_once_first = retry_once_first
        self._last_error = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._retry_once_first and len(self.calls) == 1:
            raise type("RateLimitError", (Exception,), {"status_code": 429})("slow down")
        if not self._responses:
            raise self._last_error
        payload = self._responses.pop(0)
        if isinstance(payload, Exception):
            self._last_error = payload
            raise payload
        return type("Resp", (), {"choices": [type("Ch", (), {"message": _Message(payload)})()]})()


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    def __init__(self, responses, retry_once_first=False):
        self.chat = FakeChat(FakeCompletions(responses, retry_once_first))


CLEAN = (
    "The expense ratio of HDFC Large Cap Fund – Direct Growth is 1.03%. "
    "This is the direct growth plan figure."
)

QUESTION = "What is the expense ratio of the HDFC Large Cap Fund?"


def _call_answer(client):
    return answer(QUESTION, client=client)


# --- prompt assembly ---------------------------------------------------------

def test_user_prompt_contains_chunks_and_question():
    result = retrieve(QUESTION)
    user = build_user_prompt(QUESTION, result.chunks)
    assert QUESTION in user
    for chunk in result.chunks:
        assert chunk.text in user
        assert chunk.scheme_name in user
        assert chunk.section in user


def test_system_prompt_carries_the_contract():
    assert "Answer ONLY from the CONTEXT" in prompts.SYSTEM_PROMPT
    assert prompts.DISCLAIMER in prompts.SYSTEM_PROMPT


# --- citation -----------------------------------------------------------------

def test_citation_is_the_scheme_url():
    result = retrieve(QUESTION)
    url = _citation_for(result)
    assert url == config.SOURCES[0]["url"]
    assert url in config.ALLOWED_URLS


# --- full answer path (fake client) ------------------------------------------

def test_answer_is_grounded_cited_and_valid():
    client = FakeClient([CLEAN])
    result = _call_answer(client)
    assert result.ok
    assert result.kind == "answer"
    assert result.citation == config.SOURCES[0]["url"]
    assert prompts.LAST_UPDATED_PREFIX in result.text
    check = validate_output(result.text)
    assert check.ok, check.violations


def test_answer_is_validated_before_release():
    """An answer that violates the contract becomes a safe fallback."""
    bad = (
        "You should invest in this fund now. "
        "It has returned 12% and would be ideal for your portfolio."
    )
    result = _call_answer(FakeClient([bad]))
    assert not result.ok
    assert result.kind == "fallback"
    assert prompts.NO_CONTEXT_REFUSAL in result.text
    assert result.violations, "the rejection reason must be recorded"


def test_refusals_never_call_the_llm():
    class Exploding:
        def __getattr__(self, _):
            raise AssertionError("refusal path reached the LLM client")

    result = answer("Should I invest in HDFC ELSS or HDFC Flexi Cap?", client=Exploding())
    assert result.kind == "refused"
    assert not result.ok
    assert "advice" in result.text.lower()


def test_out_of_corpus_never_calls_the_llm():
    class Exploding:
        def __getattr__(self, _):
            raise AssertionError("out-of-corpus path reached the LLM client")

    result = answer("What is the expense ratio of HDFC Mid-Cap?", client=Exploding())
    assert result.kind == "not_in_sources"
    assert not result.ok
    assert result.text.lower().startswith("i don't know")


def test_not_in_sources_has_no_number():
    """F10: a fund outside the corpus gets no figure, not even a wrong one."""
    result = answer("Can I redeem HDFC Nifty 50 Index Fund before 3 years?")
    for token in result.text.split():
        numeric = token.strip("₹%.,")
        if numeric.isdigit():
            pytest.fail(f"out-of-corpus answer leaked a number: {token!r}")


# --- retry and errors --------------------------------------------------------

def test_transient_error_is_retried_once():
    client = FakeClient([CLEAN], retry_once_first=True)
    system = prompts.SYSTEM_PROMPT
    user = build_user_prompt(QUESTION, retrieve(QUESTION).chunks)
    text = call_groq(system, user, client=client)
    assert text == CLEAN
    assert len(client.chat.completions.calls) == 2
    for call in client.chat.completions.calls:
        assert call["temperature"] == config.GROQ_TEMPERATURE
        assert call["max_tokens"] == config.GROQ_MAX_TOKENS
        assert call["model"] == config.GROQ_MODEL


def test_persistent_error_becomes_generation_error():
    client = FakeClient(
        [type("InternalServerError", (Exception,), {"status_code": 500})("boom")]
    )
    with pytest.raises(GenerationError):
        call_groq(prompts.SYSTEM_PROMPT, "question", client=client)


def test_missing_key_raises_friendly_error():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    with pytest.raises(config.MissingConfigError):
        call_groq(prompts.SYSTEM_PROMPT, "question")
    monkeypatch.undo()


def test_injected_client_does_not_need_an_api_key():
    """A test double stands in for the network, so no credential is required."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    try:
        text = call_groq(prompts.SYSTEM_PROMPT, "question", client=FakeClient([CLEAN]))
    finally:
        monkeypatch.undo()
    assert CLEAN in text


def test_failed_call_produces_a_fallback_answer():
    client = FakeClient(
        [type("InternalServerError", (Exception,), {"status_code": 500})("boom")]
    )
    result = _call_answer(client)
    assert result.kind == "error"
    assert not result.ok
    assert prompts.NO_CONTEXT_REFUSAL in result.text
    assert "Groq API error" in result.violations[0]


@pytest.mark.skipif(not __import__("os").getenv("RUN_LIVE_GROQ"), reason="set RUN_LIVE_GROQ=1 to hit the API")
def test_live_groq_answers_the_welcome_question():
    result = answer(QUESTION)
    assert result.ok, result.violations
    assert "1.03" in result.text