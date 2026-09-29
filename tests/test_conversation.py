"""Conversation-memory tests: window bounds, rewrite decision, and fallback.

These use injected fake clients, so no API key and no network are involved.
The point being locked down is that follow-up resolution is an *improvement*
that must never become a new way to fail: every path that cannot rewrite has
to hand the original question to the retriever unchanged.
"""

from __future__ import annotations

import pytest

import config
from rag import prompts
from rag.conversation import (
    ANAPHORA,
    FOLLOWUP_OPENER,
    Conversation,
    ask,
    needs_rewrite,
    resolve_question,
    rewrite_question,
)
from tests.test_generation import CLEAN, FakeClient
from rag.guardrails import check_question


CLEAN_ELSS = (
    "The lock-in period for HDFC ELSS Tax Saver Fund is 3 years. "
    "The exit load is Nil."
)

# Assembled from fragments so the literal never lands in this tracked file, the
# same way tests/test_guardrails.py does it. See that module's note on why.
PAN = "ABCDE" + "1234" + "F"


class RoutingClient(FakeClient):
    """A fake that answers the rewriter and the answerer separately.

    The two calls are told apart by their system prompt, so a test can describe
    a whole turn ("rewrite this, then answer that") in one object.
    """

    def __init__(self, rewritten=None, answer_text=CLEAN, explode=False):
        super().__init__([])
        self.rewritten = rewritten
        self.answer_text = answer_text
        self.explode = explode
        self.rewriter_calls = 0
        self.answerer_calls = 0

    def _dispatch(self, kwargs):
        system = kwargs["messages"][0]["content"]
        if system == prompts.REWRITE_SYSTEM_PROMPT:
            self.rewriter_calls += 1
            if self.explode:
                raise RuntimeError("rewriter is down")
            return self.rewritten
        self.answerer_calls += 1
        return self.answer_text


def _client(rewritten=None, answer_text=CLEAN, explode=False):
    client = RoutingClient(rewritten, answer_text, explode)
    completions = client.chat.completions

    def create(**kwargs):
        completions.calls.append(kwargs)
        payload = client._dispatch(kwargs)
        if isinstance(payload, Exception):
            raise payload
        return type("Resp", (), {"choices": [type("Ch", (), {"message": type("M", (), {"content": payload})()})()]})()

    completions.create = create
    return client


def _conversation(*turns):
    """Build a conversation from (question, answer) pairs."""
    conversation = Conversation()
    for question, answer_text in turns:
        conversation.add_user(question)
        conversation.add_assistant(answer_text)
    return conversation


# --- window bounds ------------------------------------------------------------

def test_conversation_keeps_only_the_configured_number_of_messages():
    conversation = Conversation(max_messages=10)
    assert len(conversation) == 0
    for i in range(20):
        conversation.add_user(f"question {i}")
        conversation.add_assistant(f"answer {i}")
    assert len(conversation) == 10
    assert len(conversation.messages()) == 10


def test_default_window_is_ten_messages():
    assert Conversation().max_messages == config.CONVERSATION_MESSAGES == 10


def test_window_drops_the_oldest_first():
    conversation = Conversation(max_messages=4)
    for i in range(4):
        conversation.add_user(f"q{i}")
        conversation.add_assistant(f"a{i}")
    kept = [m["content"] for m in conversation.messages()]
    assert kept == ["q2", "a2", "q3", "a3"]


def test_messages_are_returned_oldest_first_and_copied():
    conversation = _conversation(("hi", "hello"))
    messages = conversation.messages()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    messages[0]["content"] = "mutated"
    assert conversation.messages()[0]["content"] == "hi"


def test_clear_empties_the_window():
    conversation = _conversation(("q", "a"))
    conversation.clear()
    assert conversation.is_empty


# --- rewrite decision ---------------------------------------------------------

@pytest.mark.parametrize(
    "question",
    [
        "what about its fees?",
        "and what is the exit load?",
        "how about the lock-in?",
        "is it good?",
        "what is that?",
        "what are its risks?",
    ],
)
def test_followups_are_flagged_for_rewrite(question):
    assert needs_rewrite(question, _conversation(("q", "a"))) is True


@pytest.mark.parametrize(
    "question",
    [
        "What is the expense ratio of the HDFC Large Cap Fund?",
        "Who manages HDFC Balanced Advantage Fund?",
        "What is the minimum SIP amount for HDFC ELSS Tax Saver Fund?",
    ],
)
def test_self_contained_questions_are_left_alone(question):
    assert needs_rewrite(question, _conversation(("q", "a"))) is False


def test_nothing_is_rewritten_without_history():
    conversation = Conversation()
    assert needs_rewrite("what about its fees?", conversation) is False
    client = _client(explode=True)
    assert rewrite_question("what about its fees?", conversation, client=client) == (
        "what about its fees?"
    )
    assert client.rewriter_calls == 0


def test_anaphora_and_opener_patterns_are_compiled():
    assert ANAPHORA.search("what about its expense ratio")
    assert FOLLOWUP_OPENER.match("and what about the lock-in?")
    assert not FOLLOWUP_OPENER.match("What is the expense ratio?")


# --- rewrite behaviour --------------------------------------------------------

def test_followup_is_rewritten_into_a_standalone_question():
    conversation = _conversation(
        ("What is the lock-in period for HDFC ELSS Tax Saver Fund?", CLEAN_ELSS)
    )
    client = _client(rewritten="What is the exit load of HDFC ELSS Tax Saver Fund?")
    resolved = rewrite_question("what about its exit load?", conversation, client=client)
    assert resolved == "What is the exit load of HDFC ELSS Tax Saver Fund?"
    assert client.rewriter_calls == 1


def test_self_contained_question_makes_no_llm_call():
    conversation = _conversation(("q", "a"))
    client = _client(rewritten="should not be used")
    resolved = rewrite_question(
        "What is the expense ratio of the HDFC Large Cap Fund?", conversation, client=client
    )
    assert resolved == "What is the expense ratio of the HDFC Large Cap Fund?"
    assert client.rewriter_calls == 0


def test_rewriter_prompt_carries_the_history_and_the_new_question():
    prompt = prompts.build_rewrite_prompt(
        "what about its fees?",
        _conversation(("expense ratio?", "It is 1.03%.")).messages(),
    )
    assert "expense ratio?" in prompt
    assert "It is 1.03%." in prompt
    assert "LATEST QUESTION: what about its fees?" in prompt
    assert prompt.index("expense ratio?") < prompt.index("LATEST QUESTION")


def test_rewriter_prompt_strips_urls_and_the_footer():
    conversation = _conversation(
        ("q", f"The ratio is 1.03%.\n{prompts.last_updated_line('2025-01-01')}\n"
              "https://example.invalid/page")
    )
    prompt = prompts.build_rewrite_prompt("and its exit load?", conversation.messages())
    assert "https://" not in prompt
    assert prompts.LAST_UPDATED_PREFIX not in prompt
    assert "1.03%" in prompt


@pytest.mark.parametrize(
    "raw",
    [
        "",        # empty
        "   ",     # whitespace only
        "x" * 300,  # run-on commentary, not a question
    ],
)
def test_unusable_rewriter_output_falls_back_to_the_original(raw):
    conversation = _conversation(("q", "a"))
    client = FakeClient([raw])
    resolved = rewrite_question("what about its fees?", conversation, client=client)
    assert resolved == "what about its fees?"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Rewritten question: What is its expense ratio?",
         "What is its expense ratio?"),
        ('"What is its expense ratio?"', "What is its expense ratio?"),
        ("What is its expense ratio?\nThis names the scheme from earlier.",
         "What is its expense ratio?"),
    ],
)
def test_rewriter_output_is_stripped_to_a_bare_question(raw, expected):
    conversation = _conversation(("q", "a"))
    client = FakeClient([raw])
    assert rewrite_question("what about its fees?", conversation, client=client) == expected


def test_rewrite_falls_back_when_the_llm_errors():
    conversation = _conversation(("q", "a"))
    boom = type("ServerError", (Exception,), {"status_code": 500})("down")
    client = FakeClient([boom])
    assert rewrite_question("what about its fees?", conversation, client=client) == (
        "what about its fees?"
    )


def test_rewrite_survives_a_missing_api_key():
    conversation = _conversation(("q", "a"))
    client = FakeClient([CLEAN])
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    try:
        # The injected client means no key is consulted at all.
        assert rewrite_question("what about its fees?", conversation, client=client)
    finally:
        monkeypatch.undo()


# --- the whole turn -----------------------------------------------------------

def test_ask_resolves_a_followup_and_uses_the_rewritten_question_for_retrieval():
    conversation = _conversation(
        ("What is the lock-in period for HDFC ELSS Tax Saver Fund?", CLEAN_ELSS)
    )
    client = _client(
        rewritten="What is the exit load of HDFC ELSS Tax Saver Fund?",
        answer_text=CLEAN_ELSS,
    )
    result = ask(conversation, "what about its exit load?", client=client)

    assert result.question == "what about its exit load?"
    assert result.rewritten_question == "What is the exit load of HDFC ELSS Tax Saver Fund?"
    assert result.kind == "answer"
    # The rewrite is what the retriever actually saw, so it carries the filter.
    assert result.retrieved.scheme_code == "119060"
    assert client.rewriter_calls == 1
    assert client.answerer_calls == 1


def test_ask_records_the_turn_as_the_user_typed_it():
    conversation = _conversation(
        ("What is the lock-in period for HDFC ELSS Tax Saver Fund?", CLEAN_ELSS)
    )
    client = _client(
        rewritten="What is the exit load of HDFC ELSS Tax Saver Fund?",
        answer_text=CLEAN_ELSS,
    )
    ask(conversation, "what about its exit load?", client=client)

    stored = conversation.messages()
    assert [m["role"] for m in stored[-2:]] == ["user", "assistant"]
    assert stored[-2]["content"] == "what about its exit load?"
    assert CLEAN_ELSS in stored[-1]["content"]


def test_ask_records_a_followup_that_needed_no_rewrite():
    conversation = Conversation()
    client = _client(rewritten=None, answer_text=CLEAN)
    result = ask(conversation, "What is the expense ratio of HDFC Large Cap Fund?", client=client)
    assert result.rewritten_question is None
    assert result.question == "What is the expense ratio of HDFC Large Cap Fund?"
    assert client.rewriter_calls == 0
    assert len(conversation) == 2


def test_a_pii_question_is_refused_without_rewriting_and_is_not_remembered():
    conversation = _conversation(("q", "a"))
    client = _client(answer_text=CLEAN)

    result = ask(conversation, f"my PAN is {PAN}, tell me the expense ratio", client=client)

    assert result.kind == "refused"
    assert result.guardrail_kind == "pii"
    # Neither the identifier reaching the LLM nor being kept is acceptable.
    assert client.rewriter_calls == 0
    assert client.answerer_calls == 0
    assert PAN not in "".join(m["content"] for m in conversation.messages())
    assert len(conversation) == 2  # unchanged, only the pre-existing turn


def test_a_pii_followup_is_still_refused():
    conversation = _conversation(("What is the expense ratio of HDFC Large Cap Fund?", CLEAN))
    client = _client(answer_text=CLEAN)
    result = ask(conversation, f"what about its exit load? my PAN is {PAN}", client=client)
    assert result.kind == "refused"
    assert result.guardrail_kind == "pii"
    assert client.rewriter_calls == 0


def test_an_advice_refusal_is_remembered_so_the_scheme_still_resolves():
    conversation = Conversation()
    client = _client(rewritten=None, answer_text=CLEAN)
    result = ask(conversation, "Should I invest in HDFC ELSS Tax Saver Fund?", client=client)
    assert result.kind == "refused"
    assert result.guardrail_kind == "advice"
    assert len(conversation) == 2


def test_a_rewrite_failure_degrades_to_the_original_question_and_stays_honest():
    """A failed rewrite must not crash, and must not invent an answer either.

    With the rewrite unavailable the question still says "its", so retrieval has
    no scheme to filter on and the honest "not in sources" path is the correct
    outcome. What is being asserted is that the failure is absorbed.
    """
    conversation = _conversation(
        ("What is the lock-in period for HDFC ELSS Tax Saver Fund?", CLEAN_ELSS)
    )
    client = _client(rewritten=None, answer_text=CLEAN, explode=True)
    result = ask(conversation, "what about its exit load?", client=client)

    assert client.rewriter_calls == 1
    assert result.rewritten_question is None
    assert result.kind == "not_in_sources"
    assert result.retrieved.scheme_code is None
    # The answerer was never asked to invent an answer from an unresolved question.
    assert client.answerer_calls == 0
    # The turn is still remembered, so the next follow-up can be resolved.
    assert conversation.messages()[-2]["content"] == "what about its exit load?"


# --- a follow-up the topicality guard alone would refuse ----------------------

def test_an_off_topic_followup_is_rescued_by_resolving_the_reference():
    """`check_question` needs a scheme name; a follow-up has none until resolved."""
    assert check_question("what about its risk rating?").kind == "off_topic"

    conversation = _conversation(
        ("What is the risk rating of HDFC Balanced Advantage Fund?", CLEAN)
    )
    client = _client(
        rewritten="What is the risk rating of HDFC Balanced Advantage Fund?",
        answer_text=CLEAN,
    )
    result = ask(conversation, "what about its risk rating?", client=client)

    assert result.kind == "answer"
    assert result.rewritten_question == (
        "What is the risk rating of HDFC Balanced Advantage Fund?"
    )
    assert result.retrieved.scheme_code == "118968"


def test_an_off_topic_followup_stays_refused_when_the_rewrite_cannot_rescue_it():
    conversation = _conversation(
        ("What is the risk rating of HDFC Balanced Advantage Fund?", CLEAN)
    )
    # The rewriter returns the question unchanged, so nothing was resolved.
    client = _client(rewritten="what about its risk rating?", answer_text=CLEAN)
    result = ask(conversation, "what about its risk rating?", client=client)

    assert result.kind == "refused"
    assert result.guardrail_kind == "off_topic"
    assert result.rewritten_question is None
    assert client.answerer_calls == 0


def test_a_rewrite_cannot_rescue_a_question_into_a_pii_or_advice_pass():
    """The rescue re-runs the whole guardrail, so intent refusals still win."""
    conversation = _conversation(("What is the expense ratio of HDFC Large Cap Fund?", CLEAN))
    for probe in (f"what about its risk rating? my PAN is {PAN}",
                  "should I buy it?"):
        decision = check_question(probe)
        assert decision.refused
        assert decision.kind != "off_topic", probe


def test_an_advice_refusal_makes_no_rewrite_call():
    conversation = _conversation(("What is the expense ratio of HDFC Large Cap Fund?", CLEAN))
    client = _client(answer_text=CLEAN)
    result = ask(conversation, "what about its returns? should I buy it?", client=client)
    assert result.guardrail_kind == "advice"
    assert client.rewriter_calls == 0
    assert client.answerer_calls == 0


def test_allow_rewrite_false_makes_no_llm_call():
    """`--retrieval-only` must keep its promise of never calling the LLM."""
    conversation = _conversation(
        ("What is the expense ratio of HDFC Large Cap Fund?", CLEAN)
    )
    client = _client(rewritten="What is the exit load of HDFC Large Cap Fund?")
    resolved, decision = resolve_question(
        conversation, "what about its exit load?", client=client, allow_rewrite=False
    )
    assert decision.ok
    assert resolved == "what about its exit load?"
    assert client.rewriter_calls == 0


def test_allow_rewrite_false_still_refuses_a_pii_question():
    conversation = _conversation(("q", "a"))
    client = _client(answer_text=CLEAN)
    resolved, decision = resolve_question(
        conversation, f"my PAN is {PAN}", client=client, allow_rewrite=False
    )
    assert decision.kind == "pii"
    assert resolved == f"my PAN is {PAN}"
    assert client.rewriter_calls == 0
