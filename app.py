#!/usr/bin/env python3
"""Phase 6 UI — a Streamlit chat surface over the Phase 5 answer path.

    streamlit run app.py

Everything that decides *what* is shown lives in `rag/chatsession.py` and
`rag/conversation.py`; this file only draws it. That split is what lets the
compliance-relevant parts be unit-tested without a browser.

The UI adds three things to the CLI: a rendered thread, a "Sources" expander
under each answer showing the exact chunks it was grounded in, and a clear-chat
button. The answer text itself is never reformatted — the disclaimer, the
`Last updated from sources` line and the single allowlisted citation are part of
the tested answer contract.
"""

from __future__ import annotations

import streamlit as st

import config
from rag.chatsession import ChatSession, startup_problems

st.set_page_config(
    page_title="HDFC mutual fund facts",
    page_icon="📊",
    layout="centered",
)

WELCOME = (
    "Ask me about these five HDFC mutual fund schemes. I answer only from the "
    "scheme pages I have loaded, and I will say so when a fact is not in them."
)


@st.cache_resource(show_spinner="Loading the embedding model and vector store…")
def warm_up() -> None:
    """Load the encoder and the store once per server process.

    Without this, the first question pays the MiniLM load on every page refresh
    and the app looks broken for its first few seconds.
    """
    from rag.retriever import retrieve

    retrieve(config.EXAMPLE_QUESTIONS[0])


def get_session() -> ChatSession:
    """One `ChatSession` per browser tab, stored in Streamlit's session state.

    Session state dies with the tab, which is what keeps C12 (statelessness)
    true: a refresh is a new session, and nothing is written to disk.
    """
    if "session" not in st.session_state:
        st.session_state.session = ChatSession()
    return st.session_state.session


def render_sources(turn) -> None:
    """The expander under an answer: which chunks it was grounded in."""
    if not turn.has_sources:
        return
    with st.expander(f"Sources — {len(turn.sources)} chunk(s) retrieved"):
        st.caption(
            "Cosine distance is lower-is-better; the first row is the best match. "
            f"Answers are only given when a chunk lands within "
            f"{config.DISTANCE_THRESHOLD}."
        )
        for row in turn.sources:
            st.markdown(
                f"**{row.rank}. {row.scheme_name}** · `{row.section}` · "
                f"distance `{row.distance:.3f}`"
            )
            st.caption(f"fields: {row.fields}")
            with st.popover("chunk text"):
                st.write(row.text)
            st.caption(f"[{row.url}]({row.url})")


def render_turn(turn) -> None:
    """One assistant reply, with its provenance."""
    if turn.is_refusal:
        st.info(turn.text, icon="⚠️")
    else:
        st.markdown(turn.text)
    if turn.resolved_question:
        st.caption(f"Follow-up understood as: {turn.resolved_question}")
    if turn.violations and turn.kind in ("fallback", "error"):
        st.caption(turn.violations[0])
    render_sources(turn)


def render_thread(session: ChatSession) -> None:
    for message in session.messages:
        if message.role == "user":
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                render_turn(message.turn)


def handle(question: str, session: ChatSession) -> None:
    """Answer a question, showing a spinner while the model thinks.

    A free-tier cold start can take a while, and a frozen page reads as broken
    during a live demo — hence the explicit status rather than a blank wait.
    """
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the five scheme pages and drafting an answer…"):
            turn = session.ask(question)
        render_turn(turn)


def main() -> None:
    st.title("📊 HDFC mutual fund facts")

    # The disclaimer is a graded deliverable and is compared character for
    # character against the README, so it is rendered from config, not retyped.
    st.caption(f"_{config.DISCLAIMER}_")

    problems = startup_problems()
    if problems:
        for problem in problems:
            st.error(problem)
        st.stop()

    warm_up()

    session = get_session()

    with st.sidebar:
        st.markdown("**About this bot**")
        st.caption(
            "Answers come only from five HDFC scheme pages loaded at build time. "
            "No returns are reported, no recommendations are given, and questions "
            "containing personal identifiers are refused."
        )
        if st.button("Clear chat", width="stretch"):
            session.clear()
            st.rerun()
        st.divider()
        st.caption(
            f"Follow-ups are resolved using the last "
            f"{session.conversation.max_messages} messages."
        )
        st.divider()
        st.markdown("**Example questions**")
        for example in config.EXAMPLE_QUESTIONS:
            if st.button(example, key=f"ex_{example[:24]}", width="stretch"):
                handle(example, session)
                st.rerun()
        st.divider()
        st.caption(f"Model: `{config.GROQ_MODEL}` · temperature {config.GROQ_TEMPERATURE}")

    if session.is_empty:
        st.write(WELCOME)

    render_thread(session)

    typed = st.chat_input("Ask about a scheme…")
    if typed:
        handle(typed, session)
        st.rerun()


if __name__ == "__main__":
    main()
