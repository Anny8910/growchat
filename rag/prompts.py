"""All user-facing wording in one place.

Every refusal, the answer contract, the disclaimer, and the educational links
live here so the UI, the README, and `docs/sample_qa.md` cannot drift apart
(PRD P5 compares the disclaimer character for character).

This module contains no secrets and imports nothing, so it is safe to read
from tests, docs tooling, or a cold app start.
"""

from __future__ import annotations

import re

import config

DISCLAIMER = "Facts-only. No investment advice."

MAX_SENTENCES = 3
LAST_UPDATED_PREFIX = "Last updated from sources"

# C4: the 5 allowlisted pages are the only *citations*. These are educational
# pointers for refusals, which C4 explicitly permits alongside them.
EDUCATIONAL_LINKS: dict[str, str] = {
    "learn": "https://www.amfiindia.com/",
    "complaints": "https://investor.sebi.gov.in/",
}

ANSWER_CONTRACT = f"""
- Answer ONLY from the CONTEXT given. Do not use prior knowledge about any fund.
- If the CONTEXT does not contain the answer, reply exactly: I don't know.
- Never state or imply a return, CAGR, ranking, or performance comparison.
- Never recommend buying, selling, holding, or switching a fund.
- Stay within {MAX_SENTENCES} sentences.
- Do not restate the question.
""".strip()

SYSTEM_PROMPT = f"""
You are a facts-only assistant for five HDFC mutual fund scheme pages.

{config.DISCLAIMER} You report only what the source pages state. You are not a
financial adviser and you do not forecast returns.

{ANSWER_CONTRACT}
""".strip()

PII_REFUSAL = (
    "I can't help with that. I don't store or process personal identifiers such "
    "as PAN, Aadhaar, account numbers, OTPs, email addresses, or phone numbers, "
    "and I won't ask you to share one."
)

ADVICE_REFUSAL = (
    "I can't give investment advice. I only report the facts published on the "
    "five HDFC scheme pages in my sources, and I don't recommend buying, "
    "selling, or holding any fund. For guidance on choosing a fund, the "
    "educational material from AMFI is a good place to start."
)

CROSS_SCHEME_REFUSAL = (
    "I can't rank or compare the schemes, because an honest comparison would "
    "need a separate citation for every fund — which my answer format doesn't "
    "allow. Ask me about one HDFC scheme at a time and I'll report exactly "
    "what its page says."
)

OFF_TOPIC_REFUSAL = (
    "I don't know. I only answer questions about the five HDFC mutual fund "
    "schemes in my sources, and this question is outside that scope."
)

NOT_IN_SOURCES_REFUSAL = (
    "I don't know. That answer isn't in my sources, and I won't guess or use "
    "information from outside the five HDFC scheme pages I was given. You can "
    "check the official scheme page for the current figures."
)

NO_CONTEXT_REFUSAL = (
    "I don't know. I didn't find anything relevant to your question in my "
    "sources, so I have nothing factual to answer with."
)

ADVICE_LINK = EDUCATIONAL_LINKS["learn"]
OFF_TOPIC_LINK = EDUCATIONAL_LINKS["learn"]
NOT_IN_SOURCES_LINK = EDUCATIONAL_LINKS["learn"]
CROSS_SCHEME_LINK = EDUCATIONAL_LINKS["learn"]


def not_in_sources_message(scheme_url: str | None = None) -> str:
    """The "I don't know" path, pointing at the relevant page when we have one."""
    text = NOT_IN_SOURCES_REFUSAL
    if scheme_url:
        text += f"\n{scheme_url}"
    else:
        text += f"\n{EDUCATIONAL_LINKS['learn']}"
    return text


def last_updated_line(retrieved_at: str) -> str:
    return f"{LAST_UPDATED_PREFIX}: {retrieved_at}"


def build_user_prompt(question: str, chunks) -> str:
    """Assemble the retrieved chunks and the question into the user message.

    Each chunk is prefixed with the scheme name and section, so the model can
    tell which page a fact came from and has no reason to invent context.
    """
    parts = [f"CONTEXT from the five HDFC scheme pages:\n"]
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] {chunk.scheme_name} — {chunk.section} ({chunk.field_names})\n"
            f"{chunk.text}"
        )
    parts.append(
        f"\nQUESTION: {question}\n"
        f"\nAnswer the QUESTION using ONLY the CONTEXT. "
        f"Follow the system rules exactly."
    )
    return "\n\n".join(parts)


# --- Follow-up question rewriting (conversation memory) -----------------------
#
# The rewriter's ONLY job is to make a follow-up standalone. It is deliberately
# not allowed to answer, so a rewrite can never introduce a fact that was not
# already retrieved and cited downstream. If the reference cannot be resolved,
# it must return the question unchanged so the normal "I don't know" path runs.

REWRITE_SYSTEM_PROMPT = """\
You rewrite a user's follow-up question into ONE standalone question for a \
facts-only assistant that answers questions about five HDFC mutual fund \
scheme pages.

Rules:
- Use the conversation ONLY to resolve pronouns, ellipsis, and implicit \
references (for example "its", "that fund", "what about fees", "and the \
lock-in?").
- Do NOT answer the question. Do not add facts, numbers, opinions, or advice.
- Do NOT mention the conversation or explain your reasoning.
- Keep the user's intent and wording as close to the original as possible.
- If the question is already standalone, return it unchanged.
- If the conversation does not let you resolve the reference, return the \
question unchanged.
- Output only the rewritten question, on a single line, with no quotation \
marks and no prefix."""


def _history_line(text: str, limit: int = 240) -> str:
    """Compact one stored message so the rewriter sees the topic, not the noise.

    URLs and the "Last updated from sources" footer are stripped because they
    carry no referential value and would only waste context window.
    """
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(rf"{re.escape(LAST_UPDATED_PREFIX)}[^\n]*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "..."
    return text


def build_rewrite_prompt(question: str, history) -> str:
    """Assemble prior messages and the new question into the rewriter's user turn.

    `history` is the conversation so far, oldest first, as
    ``[{"role": "user" | "assistant", "content": str}, ...]``. The new question
    is passed separately so it appears last, the way the model saw it.
    """
    lines = ["CONVERSATION SO FAR (oldest first):"]
    for message in history:
        role = "User" if message.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {_history_line(message.get('content', ''))}")
    lines.append("")
    lines.append(f"LATEST QUESTION: {question}")
    lines.append("")
    lines.append(
        "Rewrite the LATEST QUESTION as one standalone question. "
        "Output only the rewritten question."
    )
    return "\n".join(lines)
