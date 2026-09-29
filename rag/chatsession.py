"""The chat session's view model — everything the UI renders, with no Streamlit.

`app.py` is deliberately thin: it draws what this module produces. Keeping the
turn-shaping here means the parts that carry the compliance rules (what is
cited, which chunks are shown, whether a refusal is labelled a refusal) are
unit-testable without a browser or a running Streamlit server.

The rules that are *not* negotiable and are asserted by the tests:

* An answer shows its sources, and a refusal does not invent any. Sources come
  from the retrieval result the generator already produced, so the expander
  cannot show a chunk the answer was not grounded in.
* The answer text is passed through verbatim. The disclaimer, the
  `Last updated from sources` line and the single citation are all part of the
  tested answer contract, and reformatting here would break it.
* A follow-up that was rewritten says so, so the user can see which question
  was actually searched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .conversation import Conversation
from .conversation import ask as conversation_ask
from .generator import Answer
from .retriever import RetrievalResult

# A refusal never has chunks behind it, so the expander is hidden rather than
# shown empty. These are the kinds that carry a reason worth displaying.
ANSWER_KIND = "answer"
REFUSAL_KINDS = ("refused", "not_in_sources", "fallback", "error")

KIND_LABELS = {
    "answer": "Answer",
    "refused": "Not answering",
    "not_in_sources": "Not in my sources",
    "fallback": "Safe fallback",
    "error": "Error",
}


@dataclass
class SourceRow:
    """One retrieved chunk, as the sources expander shows it."""

    rank: int
    scheme_name: str
    scheme_code: str
    section: str
    fields: str
    distance: float
    url: str
    text: str


@dataclass
class BotTurn:
    """One assistant reply, shaped for rendering."""

    text: str
    kind: str
    ok: bool
    citation: str | None
    sources: list[SourceRow] = field(default_factory=list)
    resolved_question: str | None = None
    guardrail_kind: str = "ok"
    violations: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, "Answer")

    @property
    def is_refusal(self) -> bool:
        return self.kind in REFUSAL_KINDS

    @property
    def has_sources(self) -> bool:
        return bool(self.sources)


def build_sources(retrieved: RetrievalResult) -> list[SourceRow]:
    """Flatten retrieved chunks into expander rows, best match first.

    `retrieve` already returns them ordered by distance, so the rank here is the
    retrieval rank — the number that matters when you want to know *why* a fact
    was or was not found.
    """
    rows = []
    for rank, chunk in enumerate(retrieved.chunks, 1):
        rows.append(
            SourceRow(
                rank=rank,
                scheme_name=chunk.scheme_name,
                scheme_code=chunk.scheme_code,
                section=chunk.section,
                fields=chunk.field_names,
                distance=chunk.distance,
                url=chunk.source_url,
                text=chunk.text,
            )
        )
    return rows


def build_turn(result: Answer) -> BotTurn:
    """Turn a generator `Answer` into the view model, without rewordings."""
    return BotTurn(
        text=result.text,
        kind=result.kind,
        ok=result.ok,
        citation=result.citation,
        # A refusal is produced before retrieval, so its result holds no chunks
        # and this stays empty by construction rather than by a special case.
        sources=build_sources(result.retrieved),
        resolved_question=result.rewritten_question,
        guardrail_kind=result.guardrail_kind,
        violations=list(result.violations),
    )


@dataclass
class Message:
    """One entry in the rendered thread."""

    role: str  # "user" | "assistant"
    content: str
    turn: BotTurn | None = None


class ChatSession:
    """A chat thread plus the bounded memory window behind it.

    The window (`Conversation`) is capped and in-process only, so a browser
    refresh starts a new session (C12). The rendered thread is unbounded but
    lives in `st.session_state` for the life of the tab, and a refusal is kept
    in it even when the memory window deliberately drops it, because the user
    still needs to see the answer they got.
    """

    def __init__(self, window: int | None = None):
        self.conversation = Conversation(max_messages=window)
        self.messages: list[Message] = []

    def __len__(self) -> int:
        return len(self.messages)

    @property
    def is_empty(self) -> bool:
        return not self.messages

    def clear(self) -> None:
        self.messages.clear()
        self.conversation.clear()

    def ask(self, question: str, client=None) -> BotTurn:
        """Answer one question, append both sides of the turn, return the reply.

        The memory window is updated inside `rag.conversation.ask`, which drops
        a PII-refused turn on purpose; the rendered thread still shows it, so
        the user sees the refusal they earned.
        """
        result = conversation_ask(self.conversation, question, client=client)
        self.messages.append(Message(role="user", content=question))
        turn = build_turn(result)
        self.messages.append(Message(role="assistant", content=result.text, turn=turn))
        return turn


def startup_problems() -> list[str]:
    """Configuration the user must fix before the chat can work at all.

    Checked up front so a missing key or an un-ingested store is a sentence in
    the UI, not a stack trace in the middle of a demo.
    """
    import config

    problems = []
    if not config.chroma_store_exists():
        problems.append(
            f"No vector store at `{config.CHROMA_DIR}`. Run `python ingest.py` first."
        )
    if not config.GROQ_API_KEY:
        problems.append(
            "`GROQ_API_KEY` is not set. Copy `.env.example` to `.env` and add "
            "a free key from https://console.groq.com/keys"
        )
    return problems
