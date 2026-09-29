"""Runs the eval query set through the guardrails only - no retrieval, no LLM.

This is the Phase 4 gate. It reads `eval/queries.txt` rather than embedding
the questions in Python, so the same file the evaluator reads is the file under
test and the two cannot drift apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.guardrails import check_question

QUERY_FILE = Path(__file__).resolve().parent.parent / "eval" / "queries.txt"


def load_queries() -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    for raw in QUERY_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        label, _, question = line.partition("|")
        queries.append((label.strip(), question.strip()))
    return queries


ALL = load_queries()
BY_LABEL = {label: [q for lab, q in ALL if lab == label] for label, _ in ALL}

# PRD §5.1 F5/F10 and Phase 4's "do not move on until".
EXPECTED_COUNTS = {
    "factual": 10,
    "advice": 6,
    "off_topic": 4,
    "out_of_corpus": 3,
}


def test_eval_file_parses():
    assert len(ALL) == sum(EXPECTED_COUNTS.values())
    for label, count in EXPECTED_COUNTS.items():
        assert len(BY_LABEL[label]) == count, f"{label}: {len(BY_LABEL[label])} != {count}"


@pytest.mark.parametrize("question", BY_LABEL["factual"])
def test_factual_questions_pass_through(question):
    """Phase 4 gate: 10/10 in-scope questions must reach retrieval."""
    decision = check_question(question)
    assert decision.ok, f"wrongly refused {question!r} as {decision.kind}"


@pytest.mark.parametrize("question", BY_LABEL["advice"])
def test_advice_questions_are_refused(question):
    decision = check_question(question)
    assert decision.refused and decision.kind == "advice", question
    assert decision.link


@pytest.mark.parametrize("question", BY_LABEL["off_topic"])
def test_off_topic_questions_are_refused(question):
    decision = check_question(question)
    assert decision.refused and decision.kind == "off_topic", question
    assert "don't know" in decision.message.lower()


@pytest.mark.parametrize("question", BY_LABEL["out_of_corpus"])
def test_out_of_corpus_questions_pass_the_guardrails(question):
    """These are in-domain but unanswerable.

    They must reach retrieval so Phase 5 can answer with "I don't know" and a
    link. Refusing them here would report a corpus gap as a bad question, and
    F10 requires a link rather than a refusal.
    """
    decision = check_question(question)
    assert decision.ok, f"out-of-corpus question blocked at the guardrail: {question!r}"


def test_eval_file_contains_no_pii():
    """PRD §5.4: a PII value stored anywhere is an explicit failure."""
    from rag.guardrails import check_pii

    for label, question in ALL:
        found = check_pii(question)
        assert not found, f"PII {found} present in the eval file: {label}"


def test_no_eval_query_embeds_a_raw_pii_literal():
    """The PAN/account/email fixtures must not leak into a tracked file.

    The needle is assembled from fragments so this assertion does not itself
    plant a PII value in a tracked source file - which would satisfy the letter
    of the rule while breaking its intent.
    """
    text = QUERY_FILE.read_text(encoding="utf-8")
    pan = "ABCDE" + "1234" + "F"
    assert pan not in text
    assert "123456" + "78901" not in text
    assert "@" not in text
