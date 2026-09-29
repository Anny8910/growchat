"""Chat-session tests: the view model behind the Streamlit UI.

`app.py` only draws, so everything worth asserting about what the user sees
lives in `rag.chatsession.py` and is tested here. Streamlit is never imported,
which keeps the suite fast and free of a browser.

The assertions that matter are the compliance ones: a refusal never shows
sources, an answer always does, and the answer text is passed through verbatim
so the tested contract (disclaimer, `Last updated` line, one citation) survives
the UI.
"""

from __future__ import annotations

import pytest

import config
from rag.chatsession import (
    BotTurn,
    ChatSession,
    Message,
    build_sources,
    build_turn,
    startup_problems,
)
from rag.generator import Answer
from rag.retriever import RetrievalResult
from tests.test_conversation import _client
from tests.test_generation import CLEAN, FakeClient

QUESTION = "What is the expense ratio of the HDFC Large Cap Fund?"

# Assembled from fragments so the literal stays out of this tracked file, the
# same way tests/test_guardrails.py does it. See that module's note on why.
PAN = "ABCDE" + "1234" + "F"


# --- sources ------------------------------------------------------------------

def test_build_sources_keeps_retrieval_order_and_carries_every_field():
    retrieved = RetrievalResult(chunks=_real_chunks(), scheme_code="119018")
    rows = build_sources(retrieved)
    assert [r.rank for r in rows] == [1, 2]
    assert rows[0].rank < rows[1].rank
    first = rows[0]
    assert first.scheme_name
    assert first.scheme_code
    assert first.section
    assert first.fields
    assert isinstance(first.distance, float)
    assert first.url in config.ALLOWED_URLS
    assert first.text


def test_no_chunks_means_no_sources():
    assert build_sources(RetrievalResult(chunks=[])) == []


# --- turns --------------------------------------------------------------------

def test_a_grounded_answer_exposes_its_sources_and_keeps_its_text():
    retrieved = RetrievalResult(chunks=_real_chunks(), scheme_code="119018", best_distance=0.2)
    result = Answer(
        question=QUESTION,
        text=CLEAN,
        citation=config.SOURCES[0]["url"],
        ok=True,
        kind="answer",
        retrieved=retrieved,
    )
    turn = build_turn(result)
    assert turn.kind == "answer"
    assert turn.is_refusal is False
    assert turn.has_sources
    assert turn.label == "Answer"
    # Verbatim: the contract's footer and citation are untouched.
    assert turn.text == CLEAN
    assert turn.text is CLEAN


@pytest.mark.parametrize("kind", ["refused", "not_in_sources", "fallback", "error"])
def test_a_refusal_never_shows_sources(kind):
    result = Answer(
        question=QUESTION,
        text="I don't know.",
        citation=None,
        ok=False,
        kind=kind,
        retrieved=RetrievalResult(chunks=[]),
    )
    turn = build_turn(result)
    assert turn.is_refusal is True
    assert turn.has_sources is False


def test_a_refusal_keeps_its_own_wording_and_violations():
    result = Answer(
        question=QUESTION,
        text="I don't know.",
        citation=None,
        ok=False,
        kind="fallback",
        retrieved=RetrievalResult(chunks=[]),
        violations=["unfinished_sentence: ..."],
    )
    turn = build_turn(result)
    assert turn.text == "I don't know."
    assert turn.violations == ["unfinished_sentence: ..."]


def test_a_rewritten_followup_is_reported_so_the_user_can_see_what_was_searched():
    result = Answer(
        question="what about its exit load?",
        text=CLEAN,
        citation=config.SOURCES[0]["url"],
        ok=True,
        kind="answer",
        retrieved=RetrievalResult(chunks=[]),
        rewritten_question="What is the exit load of HDFC Large Cap Fund?",
    )
    turn = build_turn(result)
    assert turn.resolved_question == "What is the exit load of HDFC Large Cap Fund?"


def test_an_unrewritten_answer_reports_no_resolved_question():
    result = Answer(
        question=QUESTION, text=CLEAN, citation=None, ok=True,
        kind="answer", retrieved=RetrievalResult(chunks=[]),
    )
    assert build_turn(result).resolved_question is None


def test_guardrail_kind_is_carried_through_for_display():
    result = Answer(
        question=QUESTION, text="I can't help.", citation=None, ok=False,
        kind="refused", retrieved=RetrievalResult(chunks=[]), guardrail_kind="pii",
    )
    assert build_turn(result).guardrail_kind == "pii"


def test_build_turn_copies_the_violation_list():
    violations = ["a", "b"]
    result = Answer(
        question=QUESTION, text="x", citation=None, ok=False, kind="error",
        retrieved=RetrievalResult(chunks=[]), violations=violations,
    )
    turn = build_turn(result)
    turn.violations.append("c")
    assert violations == ["a", "b"], "must not alias the generator's list"


# --- the session --------------------------------------------------------------

def test_a_new_session_is_empty():
    session = ChatSession()
    assert session.is_empty
    assert len(session) == 0


def test_asking_records_both_sides_of_the_turn():
    session = ChatSession()
    session.ask(QUESTION, client=FakeClient([CLEAN]))
    assert [m.role for m in session.messages] == ["user", "assistant"]
    assert session.messages[0].content == QUESTION
    assert session.messages[1].turn.kind == "answer"
    assert session.is_empty is False


def test_clear_empties_both_the_thread_and_the_memory_window():
    session = ChatSession()
    client = FakeClient([CLEAN])
    session.ask(QUESTION, client=client)
    session.ask("What is the exit load of HDFC Large Cap Fund?", client=client)
    assert len(session.conversation) == 4

    session.clear()
    assert session.messages == []
    assert session.conversation.is_empty


def test_the_thread_keeps_every_turn_while_the_window_stays_bounded():
    session = ChatSession(window=4)
    client = FakeClient([CLEAN] * 20)
    for _ in range(5):
        session.ask(QUESTION, client=client)
    assert len(session.messages) == 10, "the rendered thread is not truncated"
    assert len(session.conversation) == 4, "the memory window is"


def test_a_pii_turn_is_shown_but_not_remembered():
    """The user must see the refusal; the identifier must not be remembered."""
    session = ChatSession()
    client = FakeClient([CLEAN])
    session.ask(QUESTION, client=client)
    pii_question = f"my PAN is {PAN}, what is the expense ratio?"
    session.ask(pii_question, client=client)

    shown = "".join(m.content for m in session.messages)
    assert PAN in shown, "the user still sees what they typed"
    remembered = "".join(m["content"] for m in session.conversation.messages())
    assert PAN not in remembered
    assert session.messages[-1].turn.guardrail_kind == "pii"


def test_a_followup_in_a_session_is_resolved_and_shown():
    session = ChatSession()
    client = _client(
        rewritten="What is the exit load of HDFC Large Cap Fund?",
        answer_text=CLEAN,
    )
    session.ask("What is the lock-in period for HDFC ELSS Tax Saver Fund?", client=client)
    turn = session.ask("what about its exit load?", client=client)

    assert turn.resolved_question == "What is the exit load of HDFC Large Cap Fund?"
    assert session.messages[-2].content == "what about its exit load?"
    assert client.rewriter_calls == 1


def test_an_advice_refusal_is_rendered_as_a_refusal():
    session = ChatSession()
    turn = session.ask("Should I invest in HDFC ELSS?", client=FakeClient([CLEAN]))
    assert turn.is_refusal
    assert turn.guardrail_kind == "advice"
    assert turn.has_sources is False


# --- startup ------------------------------------------------------------------

def test_startup_reports_a_missing_store(monkeypatch):
    monkeypatch.setattr(config, "chroma_store_exists", lambda: False)
    problems = startup_problems()
    assert any("ingest.py" in p for p in problems)


def test_startup_reports_a_missing_api_key(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setattr(config, "chroma_store_exists", lambda: True)
    problems = startup_problems()
    assert any("GROQ_API_KEY" in p for p in problems)


def test_startup_is_quiet_when_everything_is_configured(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "a-key")
    monkeypatch.setattr(config, "chroma_store_exists", lambda: True)
    assert startup_problems() == []


def test_no_startup_problem_leaks_a_secret(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk-super-secret-value")
    monkeypatch.setattr(config, "chroma_store_exists", lambda: True)
    for problem in startup_problems():
        assert "gsk-super-secret-value" not in problem


# --- the disclaimer the UI renders --------------------------------------------

def test_the_ui_disclaimer_is_the_configured_one():
    """P5 compares this string with the README character for character."""
    assert config.DISCLAIMER == "Facts-only. No investment advice."


# --- helpers ------------------------------------------------------------------

def _real_chunks():
    """Two genuine retrieved chunks, so the tests read real metadata."""
    from rag.retriever import retrieve

    return retrieve(QUESTION).chunks[:2]
