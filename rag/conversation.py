"""In-session conversation memory and follow-up question rewriting.

The product requirement is that a follow-up like "what about its fees?" is
answered about the *same scheme* the user was just discussing. Embedding that
string on its own retrieves nothing useful, because "its" carries no scheme
name. So before retrieval we rewrite the follow-up into a standalone question
using the recent turns, and the rest of the pipeline never knows the difference.

Two properties are load-bearing and enforced here:

* **Order.** The guardrail check runs on the *original* question, before any
  rewrite call. A message carrying a PAN must never reach the LLM or the
  transcript, so the rewriter cannot be the thing that sees it first.
* **In-process only.** The buffer is a `deque` in memory and is never written
  to disk or to a log, which is what keeps C12 (statelessness) intact: a
  restart starts a fresh conversation.

The rewriter is asked only to resolve references, never to answer, so a
rewrite cannot introduce a fact that was not already retrieved and cited. When
there is no history, nothing changes; when the rewrite call fails for any
reason, the original question is used unchanged and the normal "I don't know"
path still applies.
"""

from __future__ import annotations

import re
from collections import deque

import config
from . import prompts
from .generator import Answer, GenerationError, answer as generate_answer, call_groq
from .guardrails import check_question

# A question is only worth an LLM call if it appears to point backwards. These
# are the ways a question can refer to something the previous turn mentioned
# without naming it.
ANAPHORA = re.compile(
    r"\b(?:its|it|that|this|these|those|they|them|that's|thats|"
    r"the same|one of them|there)\b",
    re.I,
)

# Openers that make a question a follow-up even without a pronoun.
FOLLOWUP_OPENER = re.compile(
    r"^\s*(?:and\s+|also,?\s+|ok(?:ay)?,?\s+|what about|how about|"
    r"what about its|what's its|whats its|what is its|what are its|"
    r"and what|and how|and is|and does|and the|and that|"
    r"is it|are they|does it|do they|can i|should i)\b",
    re.I,
)


class Conversation:
    """A bounded, in-memory window of recent messages.

    A question and its answer count as two messages, so the default window of
    `config.CONVERSATION_MESSAGES` (10) holds the last five exchanges. The
    `deque` drops the oldest message once full, so the buffer can never grow
    without bound during a long session.
    """

    def __init__(self, max_messages: int | None = None):
        self.max_messages = max_messages or config.CONVERSATION_MESSAGES
        self._messages: deque[dict] = deque(maxlen=self.max_messages)

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def is_empty(self) -> bool:
        return not self._messages

    def messages(self) -> list[dict]:
        """Recent messages, oldest first, in the shape the rewriter expects."""
        return [dict(message) for message in self._messages]

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def clear(self) -> None:
        self._messages.clear()

    def record(self, question: str, result: Answer, decision=None) -> None:
        """Store one completed turn.

        Turns refused for PII are dropped entirely: the user's message is the
        only place the identifier appears, and keeping it would defeat the
        guarantee that account numbers are neither echoed to the user nor held
        anywhere. Refusals for advice, cross-scheme, or off-topic questions are
        kept, because that text carries no identifier and keeping it lets
        follow-ups resolve to the scheme that was being discussed.
        """
        if decision is not None and decision.kind == "pii":
            return
        self.add_user(question)
        self.add_assistant(result.text)


def needs_rewrite(question: str, conversation: Conversation) -> bool:
    """Whether this question is likely to depend on earlier turns.

    Cheap lexical gate in front of the LLM call: with an empty conversation
    there is nothing to resolve, and a question that names its subject outright
    is better left alone.
    """
    if conversation.is_empty:
        return False
    if FOLLOWUP_OPENER.match(question):
        return True
    return bool(ANAPHORA.search(question))


def _clean_rewrite(raw: str, original: str) -> str:
    """Normalise the model's output, falling back to the original question.

    The rewriter is asked for a single bare line, so anything that looks like a
    label, a quotation, or commentary is discarded rather than passed on to the
    retriever as part of a "question".
    """
    text = (raw or "").strip()
    if not text:
        return original
    text = re.sub(
        r"^(?:rewritten question|standalone question|question|rewritten)\s*:\s*",
        "",
        text,
        flags=re.I,
    )
    text = text.splitlines()[0].strip().strip('"').strip("'").strip()
    if not text or len(text) > 250:
        return original
    return text


def rewrite_question(question: str, conversation: Conversation, client=None) -> str:
    """Resolve a follow-up into a standalone question.

    Returns the original question unchanged when there is no history, when the
    question does not look like a follow-up, or when the rewrite call fails.
    Failing open is deliberate: a rewrite is an improvement, not a
    precondition, and a transient LLM error must not turn a question the corpus
    can answer into a refusal.
    """
    if not needs_rewrite(question, conversation):
        return question
    try:
        raw = call_groq(
            prompts.REWRITE_SYSTEM_PROMPT,
            prompts.build_rewrite_prompt(question, conversation.messages()),
            client=client,
        )
    except (GenerationError, config.MissingConfigError):
        return question
    return _clean_rewrite(raw, question)


def resolve_question(conversation: Conversation, question: str, client=None,
                     allow_rewrite: bool = True):
    """Decide the guardrail outcome and, when allowed, the question to search.

    Returns `(resolved, decision)`. `resolved == question` means no rewrite was
    applied, and a refused `decision` means `resolved` is to be refused as it
    stands. Both `ask` and the CLI go through here so the refusal policy is
    stated once and the two paths cannot disagree.

    `allow_rewrite=False` keeps this call LLM-free, for `--retrieval-only`.
    """
    decision = check_question(question)
    if decision.refused and decision.kind != "off_topic":
        return question, decision

    candidate = (
        rewrite_question(question, conversation, client=client)
        if allow_rewrite
        else question
    )
    if candidate != question:
        rechecked = check_question(candidate)
        if rechecked.ok:
            # Rescued: the caller now gets the passing decision, not the
            # refusal that prompted the rewrite in the first place.
            return candidate, rechecked
    if decision.refused:
        return question, decision
    return candidate, decision


def ask(conversation: Conversation, question: str, client=None) -> Answer:
    """Answer one question in the context of the conversation so far.

    The guardrail runs on the *original* question before any rewrite, so a PAN
    never reaches the rewriter and the refusal needs no LLM call. PII, advice
    and cross-scheme refusals end the turn there: they are about what the user
    is asking for, and no amount of context changes that.

    Off-topic is different. `check_question` requires a question to name a
    scheme, but a follow-up like "what about its risk rating?" names none — it
    only makes sense once the pronoun is resolved. So a topicality refusal is
    the one kind that a rewrite may rescue, and only after the resolved
    question passes the *full* guardrail again. If it does not, the original
    refusal stands.

    The turn is recorded last, once there is an answer to remember.
    """
    resolved, decision = resolve_question(conversation, question, client=client)
    if decision.refused:
        result = generate_answer(question, client=client)
        conversation.record(question, result, decision)
        return result

    result = generate_answer(resolved, client=client)
    if resolved != question:
        # Keep the question the user actually typed on the answer; expose the
        # resolved form separately so the UI can show what was really searched.
        result.rewritten_question = resolved
        result.question = question
    conversation.record(question, result, decision)
    return result
