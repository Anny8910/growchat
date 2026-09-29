#!/usr/bin/env python3
"""Interactive CLI for the Phase 5 answer path.

    python ask.py                    # interactive loop
    python ask.py --retrieval-only   # show retrieval only, no LLM call
    python ask.py "the question"     # one-shot, then exit
    python ask.py --examples         # run the 3 welcome questions

In the interactive loop the session remembers the last few exchanges, so a
follow-up like "what about its fees?" is resolved to the scheme under
discussion before retrieval. Extra commands: history | reset | exit.

For every question it prints which chunks were retrieved (scheme, section,
fields, cosine distance) before the answer, so retrieval behaviour is visible
as well as the final answer.
"""

from __future__ import annotations

import argparse
import sys

import config
from rag.conversation import Conversation, ask as conversation_ask, resolve_question
from rag.generator import Answer
from rag.retriever import retrieve
from rag.retriever import RetrievalResult


def show_retrieval_result(result: RetrievalResult, retrieval_only: bool = False) -> None:
    print()
    print(f"  retrieved: {len(result.chunks)} chunk(s)"
          + (f", scheme filter {result.scheme_code}" if result.scheme_code else "")
          + (f", best distance {result.best_distance:.3f}" if result.best_distance is not None else ""))
    for i, chunk in enumerate(result.chunks, 1):
        print(
            f"    #{i} d={chunk.distance:.3f}  [{chunk.scheme_code}] "
            f"{chunk.scheme_name}\n"
            f"        section={chunk.section}  fields={chunk.field_names}"
        )
    if retrieval_only:
        print(f"    -> in_scope: {result.in_scope}")


def show_retrieval(question: str, retrieval_only: bool = False) -> None:
    show_retrieval_result(retrieve(question), retrieval_only)


def render(answer: Answer, retrieval_only: bool = False) -> None:
    label = {
        "answer": "ANSWER",
        "refused": "REFUSED",
        "not_in_sources": "NOT IN SOURCES",
        "fallback": "SAFE FALLBACK",
        "error": "ERROR",
    }[answer.kind]
    line = "  " + label + "  "
    print(line)
    print("  " + "-" * 68)
    for para in answer.text.splitlines():
        print(f"    {para}")
    if answer.kind == "fallback" and answer.violations:
        print(f"    (validate_output rejected the LLM output: {answer.violations[0]})")
    if answer.kind == "error" and answer.violations:
        print(f"    (LLM call failed: {answer.violations[0]})")
    print()


def one_question(
    conversation: Conversation,
    q: str,
    retrieval_only: bool,
    show_chunks: bool,
) -> None:
    print(f"\nQ: {q}")

    if retrieval_only:
        # Same policy as the answering path, minus the generation. Rewriting is
        # skipped so the flag keeps its promise of making no LLM call.
        resolved, decision = resolve_question(conversation, q, allow_rewrite=False)
        print(f"  guardrail: {decision.kind}"
              + (f" (pii: {decision.pii_types})" if decision.pii_types else ""))
        if decision.refused:
            print("  -> refused before retrieval; no LLM call.")
            return
        if resolved != q:
            print(f"  resolved follow-up -> {resolved}")
        if show_chunks:
            show_retrieval(resolved, retrieval_only)
        return

    result = conversation_ask(conversation, q)
    print(f"  guardrail: {result.guardrail_kind}")
    if result.rewritten_question:
        print(f"  resolved follow-up -> {result.rewritten_question}")
    if show_chunks:
        show_retrieval_result(result.retrieved)
    render(result)
    if result.refused:
        print("  (refused before retrieval; no LLM call)")
        print()


def interactive(retrieval_only: bool, show_chunks: bool) -> None:
    conversation = Conversation()
    print("HDFC facts-only assistant — Phase 5")
    print("  Type a question, or: history | reset | exit | examples")
    print(f"  Follow-ups are resolved using the last {conversation.max_messages} messages.")
    if retrieval_only:
        print("  --retrieval-only: showing retrieval only, no LLM call")
    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        lowered = raw.lower()
        if lowered in ("exit", "quit"):
            break
        if lowered in ("history", "hist"):
            print(f"  {len(conversation)} message(s) in memory:")
            for message in conversation.messages():
                who = "you" if message["role"] == "user" else "bot"
                print(f"    {who}: {message['content'].splitlines()[0][:100]}")
            continue
        if lowered in ("reset", "clear"):
            conversation.clear()
            print("  conversation cleared.")
            continue
        if lowered in ("examples", "example"):
            for example in config.EXAMPLE_QUESTIONS:
                one_question(conversation, example, retrieval_only, show_chunks)
            continue
        one_question(conversation, raw, retrieval_only, show_chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="HDFC facts-only RAG: retrieval + Groq answer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("question", nargs="*", help="one-shot question (optional)")
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="show retrieval only, skip the LLM call (no API key needed); "
             "follow-ups are not resolved in this mode",
    )
    parser.add_argument(
        "--no-chunks", action="store_true",
        help="hide the retrieved-chunk list",
    )
    parser.add_argument("--examples", action="store_true",
                        help="ask the 3 welcome-screen questions")
    args = parser.parse_args(argv)

    # A one-shot question has no preceding turns, so it gets a throwaway
    # session; only the interactive loop carries memory forward.
    if args.question:
        one_question(Conversation(), " ".join(args.question),
                     args.retrieval_only, not args.no_chunks)
        return 0
    if args.examples:
        config.print_settings()
        conversation = Conversation()
        for example in config.EXAMPLE_QUESTIONS:
            one_question(conversation, example, args.retrieval_only, not args.no_chunks)
        return 0
    interactive(args.retrieval_only, not args.no_chunks)
    return 0


if __name__ == "__main__":
    sys.exit(main())