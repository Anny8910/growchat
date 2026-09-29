# Mutual Funds Facts-Only RAG Chatbot (HDFC AMC)

A RAG chatbot that answers **factual** questions about five HDFC mutual fund schemes using only five fixed public pages. Every answer carries a source link. No investment advice, no return claims.

**Status: Phase 6 of 6 (chat UI) complete.** See [docs/implementation.md](docs/implementation.md).

---

## Scope

**AMC:** HDFC Asset Management. Five schemes, all Direct – Growth:

| Category | Scheme | `scheme_code` |
|---|---|---|
| Large Cap | HDFC Large Cap Fund – Direct Growth | `119018` |
| Flexi Cap | HDFC Equity (Flexi Cap) Fund – Direct Growth | `118955` |
| ELSS | HDFC ELSS Tax Saver Fund – Direct Plan – Growth | `119060` |
| Small Cap | HDFC Small Cap Fund – Direct Growth | `130503` |
| Balanced Advantage | HDFC Balanced Advantage Fund – Direct Growth | `118968` |

**Facts-only. No investment advice.**

---

## Requirements

- **Python 3.12** (the stack requires ≥ 3.10; the system `python3` on macOS is often 3.9, so check with `python3.12 --version`)
- A Groq API key — free tier, needed only from Phase 5 onward
- Network access on first run only, for the initial page fetch and the embedding model download (~90 MB)

---

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env      # add GROQ_API_KEY at Phase 5; not needed before then
```

Phases 1–4 need **no API key**; Phase 5 needs `GROQ_API_KEY` in `.env`.

## Running

```bash
python -c "import config; config.print_settings()"   # inspect resolved settings
python -m pytest                                        # tests
python ingest.py                                        # full: chunk → embed → store
python ingest.py --chunk-only                           # stop after chunks.txt
python ingest.py --force                                # refetch, ignoring the raw cache
python ask.py                                           # Phase 5 CLI: ask questions, see retrieved chunks
python ask.py --retrieval-only                          # retrieval only, no API key, no LLM
python ask.py --examples                                # the 3 welcome questions
streamlit run app.py                                    # Phase 6 chat UI (opens in a browser)
```

## The chat UI

```bash
streamlit run app.py
```

Then open the printed URL (usually <http://localhost:8501>).

A chat thread with the 3 example questions, a **Sources** expander under every
answer, and a **Clear chat** button in the sidebar. Two things it shows that the
terminal does not:

- **Sources.** Each answer has an expander listing the exact chunks it was
  grounded in — scheme, section, cosine distance, the fields covered, and the
  chunk text. The distances are the ones the threshold was calibrated against, so
  you can see *why* a fact was or was not found. A refusal has no expander: it is
  produced before retrieval, so there is nothing behind it.
- **The resolved follow-up.** When "what about its exit load?" is rewritten to
  name a scheme, the UI shows the resolved question, so a follow-up is never a
  black box.

The disclaimer, the `Last updated from sources` line and the single allowlisted
citation are rendered from `rag/prompts.py` and `config.py` — the UI never retypes
them, so they cannot drift from the answer contract the tests check (P5).

`app.py` only draws. Everything that decides what appears lives in
`rag/chatsession.py` and `rag/conversation.py`, which is why the compliance
behaviour is unit-testable without a browser (`tests/test_ui.py`), and the
drawing itself has headless smoke tests (`tests/test_ui_app.py`).

## Follow-up questions

In the interactive loop the session remembers the last 10 messages, so a
follow-up that relies on a pronoun is resolved before retrieval:

```
> What is the lock-in period for HDFC ELSS TaxSaver Fund?
    The lock-in period ... is 3 years.
> what about its exit load?
  resolved follow-up -> What is the exit load for HDFC ELSS TaxSaver Fund?
    The exit load ... is nil.
> and its minimum SIP amount?
  resolved follow-up -> What is the minimum SIP amount for HDFC ELSS TaxSaver Fund?
    The minimum SIP amount is ₹500.
```

Embedding "what about its exit load?" on its own retrieves nothing, because
"its" carries no scheme name. The rewriter is asked *only* to resolve
references — never to answer — so a rewrite cannot introduce a fact that was
not already retrieved and cited. `CONVERSATION_MESSAGES` sets the window
(default 10, i.e. the last five exchanges).

Three properties are deliberate:

- **In-process only.** The buffer lives in RAM and is never written to disk,
  which is what keeps C12 (statelessness) intact — a restart starts fresh.
- **The guardrail runs first.** A question carrying a PAN is refused before
  the rewriter is called, so the identifier never reaches the LLM or the
  transcript, and that turn is not remembered.
- **A rewrite is an improvement, not a precondition.** If the rewrite call
  fails, is unusable, or cannot resolve the reference, the original question is
  used unchanged and the normal "I don't know" path still applies.

One guardrail rule is relaxed by design: `check_question` normally requires a
question to name a scheme, which a follow-up never does. So a follow-up
refused as *off-topic* may be rescued by the rewrite — but only after the
resolved question passes the full guardrail again. PII, advice, and
cross-scheme refusals are never rescued.

Extra CLI commands: `history` (show the buffer), `reset` (clear it), `exit`.

## Retrieval parameters (frozen)

The Phase 3 measurements set the two numbers that govern answers, and they now
live only in `config.py` (or `.env`):

| Parameter | Value | Why |
|---|---|---|
| `TOP_K` | 5 | the Balanced Advantage risk fact sits at rank 5 of 6 |
| `DISTANCE_THRESHOLD` | 0.45 | good matches measure 0.138–0.374; off-topic 0.823+ |

A question is answered only when a chunk lands within the threshold; otherwise
the bot says "not in my sources" and links out — it never guesses (F10). The
scheme named in the question is applied as a Chroma metadata filter *before*
ranking, so a question about one fund cannot be answered from another's chunk.

## Project layout

```
config.py          every tunable + the 5-URL allowlist (the only place these live)
requirements.txt   pinned; see the note in the file about Linux/CPU-only torch
ingest.py          Phase 2+  Load → Extract → Chunk → Embed → Store
app.py             Phase 6  Streamlit chat UI (streamlit run app.py)
rag/               loader, extract, chunker, embedder, store, retriever,
                   generator, guardrails, prompts, conversation, chatsession
data/              raw HTML cache, chunks/ + embeddings_preview.txt (committed), chroma store
tests/             pytest suite
docs/              PRD, chunking strategy, architecture, implementation plan
eval/              Phase 5  query evaluation set
```

`data/chunks/chunks.txt` and `data/embeddings_preview.txt` are committed on purpose — it is a graded deliverable and must be readable without running anything. `data/raw/` and `data/chroma/` are machine-specific and ignored.

## Known limits

Documented in full in [docs/PRD.md](docs/PRD.md) §6.4. In short: only these five schemes, only the Direct – Growth variant, a point-in-time snapshot that goes stale, no live NAV, and English only.

**Resolved: `GROQ_MAX_TOKENS` was too small for the chosen model.** `openai/gpt-oss-120b`
is a *reasoning* model, and its reasoning tokens are counted against
`max_tokens`. At the original 250, the model sometimes spent the whole budget
reasoning and returned truncated text (or nothing at all), which
`validate_output` caught and answered with "I don't know" — measured 3 of 8
runs degraded, on "What is the exit load of HDFC Small Cap Fund?" at
temperature 0. The budget is now 1024, which measured 0 of 8. This never
weakened the answer contract: the ≤3-sentence cap and the single-citation rule
are enforced by `validate_output`, independently of the token budget.

## Documentation

| Doc | Contents |
|---|---|
| [docs/problemstatement.txt](docs/problemstatement.txt) | The original brief |
| [docs/PRD.md](docs/PRD.md) | Goal, users, scope, success criteria, constraints |
| [docs/CHUNKING.md](docs/CHUNKING.md) | Chunking strategy and its justification, written before any code |
| [docs/architecture.md](docs/architecture.md) | Components, data flow, tech stack, folder structure |
| [docs/implementation.md](docs/implementation.md) | The six phases and how to verify each |
