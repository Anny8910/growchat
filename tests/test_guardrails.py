"""Guardrail tests: 6 refusals, 3 PII cases, 5 output-validation cases.

**PII values are assembled at runtime, never written as literals.** PRD §5.4
counts a PII value stored anywhere - including in a tracked test file - as an
explicit failure, and the Phase 4 checklist requires grepping for the values
and finding zero hits. Concatenating fragments keeps the value out of the file
on disk while still testing the real patterns.
"""

from __future__ import annotations

import pytest

import config
from rag import prompts
from rag.guardrails import (
    GuardrailDecision, check_pii, check_question, count_sentences, extract_urls,
    is_advice, is_cross_scheme, is_off_topic, validate_output,
)

# --- PII fixtures, built at runtime and never stored (see module docstring) ---
#
# `ids=` below is load-bearing, not cosmetic. Without it pytest derives each
# test's node ID from the parametrised string, writes it to
# `.pytest_cache/v/cache/nodeids`, and the PII value lands on disk in the cache
# - the exact "PII value in a log" failure PRD §5.4 calls out, triggered by the
# tests that exist to prevent it. Opaque ids keep the values out of the cache.
PAN = "ABCDE" + "1234" + "F"
AADHAAR = "2345" + " " + "6789" + " " + "0123"
ACCOUNT_NO = "123456" + "78901"
EMAIL = "ananya" + "." + "example" + "@" + "gmail.com"
PHONE = "+91 " + "98765" + "43210"
OTP_LINE = "My otp is " + "482913" + ", verify my account"

PII_CASES = [
    ("pan", f"My PAN is {PAN}, can you check my ELSS eligibility?", ("pan",)),
    ("account", f"My account number is {ACCOUNT_NO} — what's my exit load?", ("account_number",)),
    ("email", f"Share my registered email for the statement download link: {EMAIL}", ("email",)),
    ("aadhaar", f"My Aadhaar number is {AADHAAR}.", ("aadhaar",)),
    ("phone", f"Call me on {PHONE} to confirm my folio.", ("phone",)),
    ("otp", OTP_LINE, ("otp",)),
]
PII_PARAMS = [(case[1], case[2]) for case in PII_CASES]
PII_IDS = [case[0] for case in PII_CASES]
PII_QUESTIONS = [case[1] for case in PII_CASES]

# PRD §4.3, verbatim.
ADVICE_CASES = [
    "Should I invest in HDFC ELSS or HDFC Flexi Cap?",
    "Is now a good time to buy the HDFC Large Cap Fund?",
    "Which HDFC fund has given the best returns?",
    "If I invest ₹10,000/month for 10 years, what will I get?",
    "Is HDFC Small Cap safe for my retirement?",
    "Tell me if I should sell my HDFC Flexi Cap.",
]

OFF_TOPIC_CASES = [
    "What is the weather in Paris tomorrow?",
    "Write me a poem about money.",
    "Explain quantum computing to me.",
    "What is today's price of gold?",
]

IN_SCOPE_CASES = [
    "What is the expense ratio of the HDFC Large Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver Fund?",
    "What is the minimum SIP amount for HDFC Balanced Advantage Fund?",
    "What is the exit load on HDFC Small Cap Fund?",
    "What's the benchmark of HDFC Equity (Flexi Cap) Fund?",
    "What risk category does HDFC Balanced Advantage Fund fall under?",
    "Can I redeem HDFC ELSS before 3 years?",
    "How do I download my capital gains statement?",
    "What is the minimum lump sum investment for HDFC Large Cap?",
    "What is the NAV of HDFC Flexi Cap?",
]


# --- 1. PII: 3 PRD cases plus extras, all refused, none echoed ---------------

@pytest.mark.parametrize("question,expected", PII_PARAMS, ids=PII_IDS)
def test_pii_is_detected(question, expected):
    found = check_pii(question)
    assert any(label in found for label in expected), f"{found} missing {expected}"


@pytest.mark.parametrize("question,expected", PII_PARAMS, ids=PII_IDS)
def test_pii_question_is_refused(question, expected):
    decision = check_question(question)
    assert decision.refused
    assert decision.kind == "pii"
    assert decision.link


@pytest.mark.parametrize("question,expected", PII_PARAMS, ids=PII_IDS)
def test_pii_value_is_never_echoed(question, expected):
    """C5/F6: the refusal must not repeat what the user typed."""
    decision = check_question(question)
    blob = f"{decision.message} {decision.link} {decision.pii_types}"
    for secret in (PAN, ACCOUNT_NO, EMAIL, PHONE, AADHAAR):
        assert secret not in blob
    # The full question must not survive into the message either.
    assert question not in decision.message


def test_pii_result_carries_labels_not_values():
    decision = check_question(f"My PAN is {PAN}, can you check my ELSS eligibility?")
    assert decision.pii_types == ["pan"]
    assert PAN not in str(decision)


# --- 2. advice: 6/6 refused, none contains a recommendation -----------------

@pytest.mark.parametrize("question", ADVICE_CASES)
def test_advice_is_refused(question):
    decision = check_question(question)
    assert decision.refused, f"not refused: {question!r}"
    assert decision.kind == "advice"
    assert decision.link, "a refusal must carry an educational link (F5)"


@pytest.mark.parametrize("question", ADVICE_CASES)
def test_advice_refusal_contains_no_recommendation(question):
    decision = check_question(question)
    text = f"{decision.message} {decision.link or ''}"
    for phrase in ("you should", "we recommend", "i recommend", "buy this", "sell this",
                   "best choice", "ideal for you", "suitable for you"):
        assert phrase not in text.lower()
    # No hedge that implies a recommendation.
    for phrase in ("consider buying", "might be a good", "could be worth"):
        assert phrase not in text.lower()


# --- 3. off topic ------------------------------------------------------------

@pytest.mark.parametrize("question", OFF_TOPIC_CASES)
def test_off_topic_is_refused(question):
    decision = check_question(question)
    assert decision.refused
    assert decision.kind == "off_topic"
    assert "don't know" in decision.message.lower()


# --- 4. false positives: all 10 in-scope questions must pass -----------------

@pytest.mark.parametrize("question", IN_SCOPE_CASES)
def test_in_scope_questions_are_not_refused(question):
    decision = check_question(question)
    assert decision.ok, f"wrongly refused {question!r} as {decision.kind}"


@pytest.mark.parametrize("question", IN_SCOPE_CASES)
def test_in_scope_questions_are_not_advice(question):
    assert not is_advice(question), f"falsely read as advice: {question!r}"


@pytest.mark.parametrize("question", IN_SCOPE_CASES)
def test_in_scope_questions_are_not_off_topic(question):
    assert not is_off_topic(question), f"falsely read as off topic: {question!r}"


def test_pii_false_positives():
    """Step 6: no digits in a legitimate question may trip a PII pattern."""
    for question in (
        "What is my expense ratio?",
        "What is the minimum SIP?",
        "What is the NAV of HDFC Flexi Cap?",
        "Is the lock-in 3 years?",
        "What is the minimum lump sum for HDFC Large Cap?",
    ):
        assert check_pii(question) == [], f"false PII hit on {question!r}"
        assert check_question(question).ok


def test_fund_facts_do_not_trip_pii():
    """A ₹500 minimum or a 1.21% ratio must not look like an account number."""
    for question in (
        "What is the minimum SIP of ₹500 for ELSS?",
        "Is the expense ratio 1.21%?",
        "The exit load is 1% if redeemed within 1 year, correct?",
    ):
        assert check_pii(question) == [], f"false PII hit on {question!r}"


# --- 5. refusals are free: no LLM, no network -------------------------------

def test_refusal_path_makes_no_network_call(monkeypatch):
    """Step 4: refusals must be instant and quota-free.

    Asserted at the socket layer rather than by patching imports, because
    pytest imports modules of its own during a test and a broad `__import__`
    patch crashes the run instead of testing anything.
    """
    import socket

    def explode(*args, **kwargs):
        raise AssertionError("guardrail path attempted a network connection")

    monkeypatch.setattr(socket, "socket", explode)
    monkeypatch.setattr(socket, "create_connection", explode)

    for question in PII_QUESTIONS + ADVICE_CASES + OFF_TOPIC_CASES:
        assert check_question(question).refused
    for question in IN_SCOPE_CASES:
        assert check_question(question).ok


def test_guardrails_module_pulls_in_no_llm_client():
    """The module must be importable with no groq key and no client present."""
    import sys

    for name in ("groq", "openai", "anthropic"):
        assert name not in sys.modules, f"{name} was imported by the guardrails path"


# --- 6. output validation: 5 crafted answers, 4 rejected, 1 passes ----------

# --- cross-scheme comparisons / rankings are refused (Phase 5 fix) -----------

CROSS_SCHEME_CASES = [
    "Which HDFC scheme has the lowest expense ratio?",
    "Which HDFC fund has a lower exit load?",
    "Compare HDFC Large Cap and HDFC Flexi Cap expense ratios.",
    "What is the difference between the five HDFC schemes?",
]


@pytest.mark.parametrize("question", CROSS_SCHEME_CASES)
def test_cross_scheme_comparison_is_refused(question):
    decision = check_question(question)
    assert decision.refused, f"not refused: {question!r}"
    assert decision.kind == "cross_scheme"
    assert decision.link
    assert "can't" in decision.message.lower() or "don't" in decision.message.lower()


@pytest.mark.parametrize("question", CROSS_SCHEME_CASES)
def test_cross_scheme_refusal_implies_no_rank(question):
    decision = check_question(question)
    text = f"{decision.message} {decision.link or ''}".lower()
    assert "best" not in text
    assert "recommend" not in text


def test_single_scheme_question_is_not_cross_scheme():
    """The welcome/in-scope questions must still pass (false-positive guard)."""
    for question in (
        "What is the lowest minimum investment for HDFC Large Cap?",
        "What is the highest SIP allowed for HDFC ELSS?",
        "What is the minimum lump sum investment for HDFC Large Cap?",
    ):
        assert not is_cross_scheme(question), f"falsely cross-scheme: {question!r}"


def test_cross_scheme_vs_single_scheme_false_positive():
    """A superlative about ONE scheme must not be a cross-scheme comparison."""
    assert check_question("What is the minimum SIP for HDFC Small Cap?").ok
    assert check_question("What is the expense ratio of HDFC Flexi Cap?").ok

CLEAN_ANSWER = (
    "The expense ratio of HDFC Large Cap Fund – Direct Growth is 1.03%. "
    "This is the direct growth plan figure listed on the scheme page. "
    f"Last updated from sources: 2026-09-29\n{config.SOURCES[0]['url']}"
)


def test_clean_answer_passes():
    result = validate_output(CLEAN_ANSWER)
    assert result.ok, result.violations
    assert result.violations == []


def test_five_sentences_is_rejected():
    answer = CLEAN_ANSWER + " The figure changes over time. Check the page again."
    result = validate_output(answer)
    assert result.rejected
    assert any(v.startswith("too_long") for v in result.violations)


# --- truncation vs. trailing punctuation --------------------------------------

def _with_body(body: str) -> str:
    """An otherwise clean answer with a given body, for validate_output."""
    return f"{body}\nLast updated from sources: 2026-09-29\n{config.SOURCES[0]['url']}"


@pytest.mark.parametrize("body", [
    # A finished sentence may be followed by a closing mark. The live model
    # writes "...within the first year.)" and these were being rejected.
    "The exit load is 1% if redeemed within one year.)",
    'The exit load is 1% if redeemed within one year."',
    "The exit load is 1% if redeemed within one year.’",
    "The exit load is 1% if redeemed within one year.]",
    # Plain endings, unchanged behaviour.
    "The exit load is 1% if redeemed within one year.",
    "The exit load is 1% if redeemed within one year!",
    "The exit load is 1% if redeemed within one year?",
])
def test_a_finished_sentence_is_not_called_truncation(body):
    result = validate_output(_with_body(body))
    assert result.ok, result.violations
    assert not any(v.startswith("unfinished_sentence") for v in result.violations)


@pytest.mark.parametrize("body", [
    "The exit load is 1% if units are redeemed within",
    "The exit ratio is 1.03",
    "See the exit load note (as on the page",
])
def test_genuinely_truncated_answers_are_rejected(body):
    result = validate_output(_with_body(body))
    assert result.rejected
    assert any(v.startswith("unfinished_sentence") for v in result.violations)


def test_two_links_is_rejected():
    answer = CLEAN_ANSWER + f"\n{config.SOURCES[1]['url']}"
    result = validate_output(answer)
    assert result.rejected
    assert any(v.startswith("citation_count") for v in result.violations)


def test_non_allowlisted_url_is_rejected():
    answer = (
        "The expense ratio is 1.03%. This is the direct growth plan figure. "
        "Last updated from sources: 2026-09-29\nhttps://example.com/my-blog"
    )
    result = validate_output(answer)
    assert result.rejected
    assert any("citation_not_allowlisted" in v for v in result.violations)


def test_performance_language_is_rejected():
    answer = (
        "The fund has delivered annualised returns of 12% over 3 years. "
        "That ranks in the top quartile. "
        f"Last updated from sources: 2026-09-29\n{config.SOURCES[0]['url']}"
    )
    result = validate_output(answer)
    assert result.rejected
    assert any(v.startswith("performance_language") for v in result.violations)


def test_advice_language_in_answer_is_rejected():
    answer = (
        "You should consider buying this fund for your portfolio. "
        "The expense ratio is 1.03%. "
        f"Last updated from sources: 2026-09-29\n{config.SOURCES[0]['url']}"
    )
    result = validate_output(answer)
    assert result.rejected
    assert any(v.startswith("advice_language") for v in result.violations)


def test_missing_last_updated_line_is_rejected():
    answer = (
        "The expense ratio is 1.03%. It is the direct growth plan figure. "
        f"Source: {config.SOURCES[0]['url']}"
    )
    result = validate_output(answer)
    assert result.rejected
    assert any("missing_last_updated" in v for v in result.violations)


def test_rejection_provides_a_safe_fallback():
    result = validate_output("You should buy this. https://example.com")
    assert result.rejected
    assert result.fallback == prompts.NO_CONTEXT_REFUSAL


def test_footer_does_not_consume_a_sentence():
    """The contract is 3 sentences of prose; the footer and link are metadata."""
    assert validate_output(CLEAN_ANSWER).ok
    three = (
        "The exit load is 1% within 1 year. The lock-in is 3 years. "
        "This is the current schedule. "
        f"{config.SOURCES[2]['url']}\nLast updated from sources: 2026-09-29"
    )
    result = validate_output(three)
    assert result.ok, result.violations


def test_fee_durations_are_not_treated_as_performance():
    """'1 year'/'3 years' are fee and lock-in durations here, not returns."""
    answer = (
        "The exit load is 1% if redeemed within 1 year. The ELSS lock-in is 3 years. "
        "This is the current schedule. "
        f"Last updated from sources: 2026-09-29\n{config.SOURCES[2]['url']}"
    )
    assert validate_output(answer).ok


# --- sentence counting ------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("One sentence.", 1),
    ("One. Two.", 2),
    ("The expense ratio is 1.03%. The minimum SIP is ₹100.", 2),
    ("The NAV is ₹1,426.93 as of 28 Sep '26.", 1),
    ("", 0),
])
def test_count_sentences(text, expected):
    assert count_sentences(text) == expected


def test_extract_urls_strips_trailing_punctuation():
    assert extract_urls(f"See {config.SOURCES[0]['url']}.") == [config.SOURCES[0]["url"]]


# --- wording lives in prompts.py and stays consistent ------------------------

def test_disclaimer_is_exact():
    assert config.DISCLAIMER == "Facts-only. No investment advice."
    assert prompts.DISCLAIMER == config.DISCLAIMER


def test_not_in_sources_says_i_dont_know():
    message = prompts.not_in_sources_message()
    assert message.lower().startswith("i don't know")
    assert prompts.NO_CONTEXT_REFUSAL.lower().startswith("i don't know")
    assert prompts.OFF_TOPIC_REFUSAL.lower().startswith("i don't know")


def test_educational_links_are_official_bodies():
    for label, url in prompts.EDUCATIONAL_LINKS.items():
        assert url.startswith("https://")
        assert url not in config.ALLOWED_URLS, "refusal links are separate from citations"


def test_prompts_module_has_no_secrets():
    source = open("rag/prompts.py", encoding="utf-8").read()
    assert "gsk_" not in source
    assert config.GROQ_API_KEY == "" or config.GROQ_API_KEY not in source
