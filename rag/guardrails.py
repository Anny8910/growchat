"""Refusals and output checks. Pure logic, no I/O, no API key, no LLM.

Three gates, in priority order:

1. `check_pii`  - personal identifiers are refused and never echoed (C5/F6)
2. `is_advice`  - no buy/sell/hold, timing, or return-chasing (F5/F7)
3. `is_off_topic` - anything that is not about the five scheme pages

Plus `validate_output`, which re-checks an answer the LLM has already produced
so a contract violation becomes a safe fallback instead of bad text on screen.

**The false-positive problem.** A guardrail that refuses "What is the expense
ratio?" is not a working demo. Every pattern below is therefore anchored on a
full phrase rather than a bare keyword: there is no bare `best`, `should`, or
`returns` pattern in this file, because "should" and "best" both appear in
legitimate questions. `tests/test_guardrails.py` pins the 10 in-scope
questions from PRD §4.1-4.2 as must-pass.

**PII values are never returned, logged, or stored.** Detection results carry
only a label like "pan", never the matched text, so a caller cannot leak it by
accident. PRD §5.4 treats a PII value in a log line as an automatic failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import config
from . import prompts


@dataclass
class GuardrailDecision:
    """Outcome of a pre-LLM check. `ok=True` means "safe to answer"."""

    ok: bool
    kind: str = "ok"
    message: str = ""
    link: str | None = None
    pii_types: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return not self.ok


# --------------------------------------------------------------------------
# 1. PII
# --------------------------------------------------------------------------

# Longest-first ordering matters: the 12-digit Aadhaar pattern must win over the
# generic long-digit account pattern when both could match.
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("pan", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    ("aadhaar", re.compile(r"\b[2-9][0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b")),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"(?:\+?91[-\s]?)?\b[6-9][0-9]{9}\b")),
    ("account_number", re.compile(r"\b[0-9]{9,}\b")),
    # An OTP is only detectable with its label, since 4-6 bare digits occur
    # legitimately in fund facts (a ₹500 minimum, a 1.21% ratio).
    ("otp", re.compile(r"\b(?:otp|one[-\s]?time\s+pass(?:word|code))\b\s*(?:is|:)?\s*[0-9]{4,6}\b", re.I)),
]


def check_pii(text: str) -> list[str]:
    """Return the *types* of personal identifier present. Never the values."""
    found: list[str] = []
    for label, pattern in PII_PATTERNS:
        if pattern.search(text):
            found.append(label)
    return found


# --------------------------------------------------------------------------
# 2. Advice
# --------------------------------------------------------------------------

# Each pattern is a full phrase. There is deliberately no bare "should",
# "best", "buy", "sell", or "returns" pattern here.
ADVICE_PATTERNS: list[re.Pattern[str]] = [
    # explicit recommendation requests
    re.compile(r"\bshould\s+(?:i|we|you|my|our)\b", re.I),
    re.compile(r"\b(?:i|we)\s+should\b", re.I),
    re.compile(r"\b(?:can|could|would)\s+you\s+recommend\b", re.I),
    re.compile(r"\brecommend(?:ation)?\b", re.I),
    re.compile(r"\bwhat\s+do\s+you\s+think\b", re.I),
    re.compile(r"\bgive\s+me\s+your\s+(?:opinion|view)\b", re.I),
    re.compile(r"\bworth\s+(?:investing|buying|putting)\b", re.I),
    re.compile(r"\bhelp\s+me\s+choose\b", re.I),
    re.compile(r"\bwhich\s+(?:one\s+)?(?:should|is\s+better)\b", re.I),
    re.compile(r"\bwhich\s+(?:fund|scheme|one)\b.*\bbest\b", re.I),
    # timing
    re.compile(r"\b(?:a|the)?\s*good\s+time\s+to\b", re.I),
    re.compile(r"\bright\s+time\s+to\b", re.I),
    re.compile(r"\b(?:is|are)\s+now\s+(?:a\s+)?(?:good|right|ideal)\s+time\b", re.I),
    # performance chasing
    re.compile(r"\b(?:best|highest|top|maximum)\s+(?:returns?|performing|performer|gains?)\b", re.I),
    re.compile(r"\breturns?\b", re.I),
    re.compile(r"\b(?:given|generate[ds]?|delivered|offered|yield(?:ed|s)?)\b[^.?]{0,40}\bpercent\b", re.I),
    # projections
    re.compile(r"\bwhat\s+will\s+(?:i|we|my|our)\s+(?:get|earn|make|receive)\b", re.I),
    re.compile(r"\bhow\s+much\s+will\s+(?:i|we|my)\b", re.I),
    re.compile(r"\bafter\s+\d+\s+years?\b[^.?]{0,30}\bwhat\b", re.I),
    re.compile(r"\bproject(?:ion|ed)?\b", re.I),
    # personal suitability
    re.compile(r"\b(?:safe|suitable|appropriate|good|ideal)\s+for\s+(?:my|our|your)\b", re.I),
    re.compile(r"\b(?:should|is)\b[^.?]{0,60}\bfor\s+my\s+retirement\b", re.I),
    re.compile(r"\b(?:my|our)\s+(?:retirement|portfolio\s+allocation|goal)\b", re.I),
    # action verbs aimed at the user's own money
    re.compile(r"\b(?:buy|sell|purchase|switch|hold|exit|book|allocate)\s+(?:it|them|this|that)\b", re.I),
    re.compile(r"\bstart\s+investing\b", re.I),
]


def is_advice(text: str) -> bool:
    return any(pattern.search(text) for pattern in ADVICE_PATTERNS)


# --------------------------------------------------------------------------
# 3. Cross-scheme comparison / ranking
# --------------------------------------------------------------------------

# A question that asks to rank or compare the five schemes can only be answered
# honestly with one citation per scheme, but the answer contract enforces
# exactly one URL. Rather than let the model tile shut or imply one page covers
# another fund's figure, refuse before the LLM is called (quota-free, F10).
#
# These patterns must not fire on single-scheme questions: "Which HDFC fund
# performs best?" is already caught as advice above, and every in-scope PRD
# question either names a scheme or asks about one scheme in general terms.
CROSS_SCHEME_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwhich\s+(?:one\s+)?(?:of\s+)?(?:the\s+)?(?:hdfc\s+)?(?:scheme|schemes|fund|funds|mutual\s+fund)\b", re.I),
    re.compile(r"\b(?:compare|comparison|difference\s+between|versus|\bvs\.?)\b", re.I),
]


def is_cross_scheme(text: str) -> bool:
    return any(pattern.search(text) for pattern in CROSS_SCHEME_PATTERNS)


# --------------------------------------------------------------------------
# 4. Off topic
# --------------------------------------------------------------------------

# Scope vocabulary taken from the corpus itself: if a question shares no term
# with the five scheme pages, it cannot be answered from them.
#
# Scope is a *domain* test, not a coverage test. A question can be squarely
# about mutual funds and still be unanswerable from these five pages, e.g.
# "How do I download my capital gains statement?" (PRD §4.2). That must pass
# the guardrails and fall through to retrieval, which answers it honestly with
# "I don't know" - it must not be refused here as off topic, because refusing
# it would misdescribe a gap in the corpus as a gap in the user's question.
SCOPE_TERMS: tuple[str, ...] = (
    "hdfc", "mutual fund", "mutual funds", "mf", "sip", "nav", "expense ratio",
    "exit load", "lock-in", "lock in", "lockin", "benchmark", "elss", "lump sum",
    "lumpsum", "folio", "scheme", "amc", "fund", "80c", "tax saver",
    "balanced advantage", "large cap", "flexi cap", "small cap", "mid cap",
    "mid-cap", "dividend", "idcw", "direct plan", "growth plan", "ter",
    "sebi", "amfi", "switching", "redeem", "invest",
    "capital gains", "tax statement", "statement", "cgt", "demat",
    "portfolio", "holding", "nav aid",
)


def is_off_topic(text: str) -> bool:
    """True when the question shares no vocabulary with the five scheme pages."""
    lowered = text.lower()
    return not any(term in lowered for term in SCOPE_TERMS)


# --------------------------------------------------------------------------
# 4. Output validation
# --------------------------------------------------------------------------

URL_RE = re.compile(r"https?://[^\s)\]<>\"']+")

# Marks that may sit after the full stop of a finished sentence.
CLOSING_MARKS = ")]}\"'*”’»›"

# Performance language. Deliberately excludes bare "1 year"/"3 years", which
# are legitimate fee and lock-in durations in this corpus (see Phase 2 step 3).
PERFORMANCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bannuali[sz]ed\b", re.I),
    re.compile(r"\bcagr\b", re.I),
    re.compile(r"\bcategor(?:y|ies)\s+average\b", re.I),
    re.compile(r"\brank(?:s|ed|ing)?\b", re.I),
    re.compile(r"\bbest\s+performing\b", re.I),
    re.compile(r"\b(?:historic(?:al)?|past)\s+(?:returns?|performance)\b", re.I),
    re.compile(r"\breturns?\s+of\b", re.I),
    re.compile(r"\baverage\s+returns?\b", re.I),
    re.compile(r"[0-9][0-9.]*\s*%\s*returns?\b", re.I),
    re.compile(r"\bupside\b|\bdownside\b", re.I),
]

# Recommendation language in an answer the LLM produced.
ANSWER_ADVICE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\byou\s+should\b", re.I),
    re.compile(r"\b(?:i|we)\s+(?:would\s+)?recommend\b", re.I),
    re.compile(r"\bmy\s+recommendation\b", re.I),
    re.compile(r"\bconsider\s+(?:buying|investing|selling|switching)\b", re.I),
    re.compile(r"\bworth\s+(?:investing|buying)\b", re.I),
    re.compile(r"\bbest\s+time\s+to\b", re.I),
    re.compile(r"\b(?:safe|suitable|appropriate)\s+for\s+(?:my|your|our)\b", re.I),
    re.compile(r"\bwe\s+advise\b", re.I),
]


@dataclass
class OutputCheck:
    ok: bool
    violations: list[str] = field(default_factory=list)
    fallback: str = ""

    @property
    def rejected(self) -> bool:
        return not self.ok


def answer_body(text: str) -> str:
    """Strip the citation and the "Last updated" footer before counting.

    Neither is prose. Counting the footer as a sentence would quietly shrink
    the contract from 3 sentences to 2 and reject answers that obey it.
    """
    body = re.sub(rf"{re.escape(prompts.LAST_UPDATED_PREFIX)}[^\n]*", "", text, flags=re.I)
    body = URL_RE.sub("", body)
    return body.strip()


def count_sentences(text: str) -> int:
    """Sentence count that does not split on decimals.

    `1.03%` and `₹1,426.93` must stay inside one sentence, so the split only
    happens after terminal punctuation *followed by whitespace*.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    parts = [p for p in re.split(r"(?<=[.!?])\s+", stripped) if p.strip()]
    return len(parts)


def extract_urls(text: str) -> list[str]:
    return [url.rstrip(".,;)") for url in URL_RE.findall(text)]


def validate_output(answer: str) -> OutputCheck:
    """Check a generated answer against the contract. Never raises."""
    violations: list[str] = []

    sentences = count_sentences(answer_body(answer))
    if sentences > prompts.MAX_SENTENCES:
        violations.append(f"too_long: {sentences} sentences (max {prompts.MAX_SENTENCES})")

    body = answer_body(answer)
    # A finished sentence can be followed by a closing bracket or quote — the
    # model writes "...within the first year.)" — so the check is about the
    # sentence ending, not about the last character in the string.
    trimmed = body.rstrip().rstrip(CLOSING_MARKS)
    if trimmed and not trimmed.endswith((".", "!", "?")):
        violations.append(
            f"unfinished_sentence: the answer ends without final punctuation "
            f"({trimmed[-40:]!r}) — likely truncation"
        )

    urls = extract_urls(answer)
    if len(urls) != 1:
        violations.append(f"citation_count: {len(urls)} links (exactly 1 required)")
    else:
        if urls[0] not in config.ALLOWED_URLS:
            violations.append(f"citation_not_allowlisted: {urls[0]}")
        for foreign in set(urls) - config.ALLOWED_URLS - set(prompts.EDUCATIONAL_LINKS.values()):
            violations.append(f"unexpected_link: {foreign}")

    if prompts.LAST_UPDATED_PREFIX.lower() not in answer.lower():
        violations.append("missing_last_updated_line")

    for pattern in PERFORMANCE_PATTERNS:
        hit = pattern.search(answer)
        if hit:
            violations.append(f"performance_language: {hit.group(0)!r}")
            break

    for pattern in ANSWER_ADVICE_PATTERNS:
        hit = pattern.search(answer)
        if hit:
            violations.append(f"advice_language: {hit.group(0)!r}")
            break

    if not violations:
        return OutputCheck(ok=True)

    return OutputCheck(
        ok=False,
        violations=violations,
        fallback=prompts.NO_CONTEXT_REFUSAL,
    )


# --------------------------------------------------------------------------
# The single entry point used by the request path
# --------------------------------------------------------------------------

def check_question(question: str) -> GuardrailDecision:
    """Decide whether a question may be answered. Makes no network or LLM call."""
    pii_types = check_pii(question)
    if pii_types:
        return GuardrailDecision(
            ok=False,
            kind="pii",
            message=prompts.PII_REFUSAL,
            link=prompts.EDUCATIONAL_LINKS["learn"],
            pii_types=pii_types,
        )

    if is_advice(question):
        return GuardrailDecision(
            ok=False,
            kind="advice",
            message=prompts.ADVICE_REFUSAL,
            link=prompts.ADVICE_LINK,
        )

    if is_cross_scheme(question):
        return GuardrailDecision(
            ok=False,
            kind="cross_scheme",
            message=prompts.CROSS_SCHEME_REFUSAL,
            link=prompts.CROSS_SCHEME_LINK,
        )

    if is_off_topic(question):
        return GuardrailDecision(
            ok=False,
            kind="off_topic",
            message=prompts.OFF_TOPIC_REFUSAL,
            link=prompts.OFF_TOPIC_LINK,
        )

    return GuardrailDecision(ok=True)
