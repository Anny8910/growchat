# Implementation Plan — Mutual Funds Facts-Only RAG Chatbot

**Version:** 1.0
**Date:** 2026-09-29
**Derived from:** `docs/architecture.md` §9, `docs/CHUNKING.md`
**Status:** Ready to build

Six phases, ordered so that each one is provable before the next begins. **Phases 1–3 need no API key** — the entire retrieval pipeline is verifiable before the LLM is ever called.

---

## 0. Phase overview

| Phase | Name | Needs API key? | Produces |
|---|---|---|---|
| 1 | Project setup | No | Venv, pinned deps, config skeleton |
| 2 | Loading & chunking | No | `data/raw/*.html` + `*.txt`, `data/chunks/chunks.txt` |
| 3 | Embedding & vector store | No | `data/chroma/`, `data/embeddings_preview.txt` |
| 4 | Guardrails | No | Refusal + PII logic, testable standalone |
| 5 | Retrieval + LLM answer | **Yes** (Groq) | Working text answers with citations |
| 6 | UI | Yes (inherited) | The demo |

```
1 ──► 2 ──► 3 ──► 4 ──► 5 ──► 6
setup  data   store  guard  answer  UI
```

**Verification philosophy:** every phase ends with a check you can run and a *known expected value*. "It ran" is not verification — "it produced 42 chunks, none containing a performance word, and the ELSS lock-in chunk reads 3 years" is.

**Ground truth used below** was re-read from the live pages during Phase 2 (2026-09-29). These values change; re-verify against the source page rather than trusting this document, and treat a mismatch as "the page changed" until proven otherwise.

| Scheme | `scheme_code` | Expense ratio | Min SIP | Benchmark | Risk (page prose) |
|---|---|---|---|---|---|
| Large Cap | `119018` | 1.03% | ₹100 | NIFTY 100 Total Return Index | Very High |
| Flexi Cap | `118955` | 0.77% | ₹100 | NIFTY 500 Total Return Index | Very High |
| ELSS | `119060` | 1.21% | ₹500 | NIFTY 500 Total Return Index | Very High |
| Small Cap | `130503` | 0.78% | ₹100 | BSE 250 SmallCap Total Return Index | Very High |
| Balanced Advantage | `118968` | 0.78% | ₹100 | NIFTY 50 Hybrid Composite Debt 50:50 Index | Very High |

Exit load: Large Cap / Flexi Cap 1% within 1 year; Small Cap 1% within 1 year; ELSS Nil; Balanced Advantage 1% within 1 year on units in excess of 15% of the investment. ELSS lock-in 3 years.

> **Two source inconsistencies found in Phase 2 — do not "fix" the extractor for these.**
> 1. The "Min. for SIP" figure is ₹100 on four schemes, not ₹500. An earlier draft of this table said ₹500 across the board; the pages say otherwise and the extractor follows the pages.
> 2. Every scheme's "About" prose quotes the same AUM (₹9,86,237 Cr) and a stale NAV, which contradicts the header values in the same page. The extractor drops those prose sentences and takes AUM/NAV from the header only, so a chunk can never contain two conflicting numbers.

---

## Phase 1 — Project setup

**Goal:** a runnable Python project skeleton with pinned dependencies and a single source of truth for configuration.

### Files to create

| File | Purpose |
|---|---|
| `requirements.txt` | Pinned dependencies. Pinning is load-bearing, not tidiness — see Phase 3 |
| `.gitignore` | Excludes `.env`, `data/chroma/`, `data/raw/`, `__pycache__/`, virtualenv. Explicitly does **not** exclude `data/chunks/` |
| `.env.example` | Template listing `GROQ_API_KEY` and the optional overrides, with no real value |
| `config.py` | The 5 source URLs (slug, scheme name, category, URL), embedding model id, `CHROMA_DIR`, `CHUNK_WORDS`, `CHUNK_OVERLAP`, `TOP_K`, `DISTANCE_THRESHOLD`, `GROQ_MODEL`, `GROQ_API_KEY`. Env-overridable with defaults. **The URL allowlist lives here and nowhere else** |
| `rag/__init__.py` | Empty package marker |
| `tests/__init__.py` | Empty package marker |
| `README.md` (stub) | Setup steps and scope; full version in the final pass |

### What this phase does

- Establishes a virtual environment and installs pinned packages: `requests`, `beautifulsoup4`, `lxml`, `chromadb`, `sentence-transformers`, `fastapi`, `uvicorn`, `python-dotenv`, `pytest` (and the Groq client at Phase 5).
- Centralises every tunable. `config.py` is the only place a chunk size, a threshold, or a URL is written down — so when retrieval misbehaves in Phase 5, there is exactly one file to open.
- Validates configuration at import time and fails with a readable setup message when `GROQ_API_KEY` is missing, rather than a stack trace at the first request.

### How to verify

1. Create the venv, install from `requirements.txt`, and confirm the import of every pinned dependency succeeds.
2. Start Python and import `config` — it should print the resolved settings and the 5 URLs. Confirm there are exactly 5 and they match PRD §3.1.
3. Confirm a missing `GROQ_API_KEY` produces the friendly setup message, not a traceback.
4. Temporarily remove `GROQ_API_KEY` from `.env`, re-run, then restore it.
5. `git status` — confirm `.env` and `data/` are ignored while `docs/`, `config.py`, and `requirements.txt` are not.
6. Run `pytest` — it collects zero tests and exits cleanly. A collection *error* here means a path is wrong and will bite in Phase 2.

**Do not move on until:** all imports succeed, config prints the 5 URLs, and the missing-key message is clean.

**Watch for:** `chromadb` and `sentence-transformers` have heavy transitive dependencies (notably `onnxruntime` on some platforms). A slow or large install is expected, not a failure. Confirm the install completes before debugging anything else.

---

## Phase 2 — Loading & chunking

**Goal:** turn the 5 URLs into a clean, compliance-safe, human-inspectable `chunks.txt`. This is the highest-risk phase and the one the evaluator will scrutinise, because it is where the corpus is built.

### Files to create

| File | Purpose |
|---|---|
| `rag/loader.py` | Fetch with a browser `User-Agent`; non-200 or an undersized body raises naming the URL; writes `data/raw/<slug>.html` and `<slug>.txt`; records `fetched_at`; cache hit skips the network |
| `rag/extract.py` | Section allowlist keyed on visible heading text → `FactRecord` objects. Contains the scheme-code guard, the performance-region exclusion, and the `nfo_risk` denylist |
| `rag/chunker.py` | Re-joins label/value pairs into self-describing text; packs per section; builds metadata; writes `data/chunks/chunks.txt` |
| `ingest.py` | CLI entry point. Phase 2 runs it in a **load-and-chunk only** mode so the store is not built yet |
| `tests/test_extraction.py` | Asserts no performance fields, no cross-scheme records, no `nfo_risk` |
| `tests/test_chunking.py` | Asserts size cap, one scheme per chunk, no field split from its label |

### What this phase does

- Fetches all 5 pages and caches the raw HTML, so later phases work offline and the "Last updated" date comes from the cache rather than being guessed.
- Extracts **only allowlisted sections** — scheme overview, benchmark, exit load, fees, ELSS lock-in, tax/stamp duty, statement guide. Everything not on the list is invisible by design.
- Applies the scheme-code guard: a fact is admitted only if it belongs to the page's own scheme. This is what stops the Balanced Advantage page (82 embedded scheme codes) from answering questions about other funds.
- Excludes returns tables, CAGR, category rankings and averages, and the `alpha` / `c_return` / `p_return` records.
- Re-joins the label/value pairs the page splits across lines, so a retrieved chunk reads as a self-contained statement rather than a bare number.
- Caps chunks at 150 words (~200 tokens, under MiniLM's 256 limit), applies 25-word overlap to prose only, keeps one scheme and one topic per chunk, and writes every chunk with full metadata to `data/chunks/chunks.txt`.

### How to verify

1. **Fetch.** Run `ingest.py` in load-and-chunk mode. Expect exactly 5 successful fetches and 5 files in `data/raw/`. Re-run with the network off — it must complete from cache.
2. **Count.** Read `data/chunks/chunks.txt`. Expect roughly **25–35 chunks**, 5–7 per page. Open the file and read every chunk by eye. This file is a graded deliverable (F9) — the quality bar is that a human can read it and find a fact without a parser.
3. **Compliance check (the important one).** Search the whole of `data/chunks/chunks.txt` for `annualised`, `annualized`, `CAGR`, `returns`, `rank`, `category average`, `alpha`, `historic`. **Expect zero hits.** Any hit means an exclusion rule has leaked, and it must be fixed before Phase 3 — once it is embedded, the vector store carries the claim permanently.
   *Note: `1 year` / `3 year` do legitimately appear, inside exit-load slabs ("Exit load of 1% if redeemed within 1 year") and the ELSS lock-in ("3 years"). Those are fee and lock-in durations, not return claims — do not treat them as leaks.*
4. **Cross-scheme check.** Confirm every chunk's `scheme_code` is one of `119018`, `118955`, `119060`, `130503`, `118968`, and that no chunk from the Large Cap page mentions a Small Cap figure. Read the Balanced Advantage chunks specifically — it is the most contaminated page.
5. **Boilerplate check.** Search for `Moderately High` and `nfo_risk`. Expect zero hits — the real risk reading is "Very High" from the page prose, and the boilerplate contradicts it.
6. **Spot-check five facts against the source page**, side by side in a browser:
   - ELSS lock-in → `3 years`
   - ELSS expense ratio → `1.21%`
   - ELSS minimum SIP → `₹500`
   - Large Cap exit load → `1% if redeemed within 1 year`
   - ELSS benchmark → `NIFTY 500 Total Return Index`
   A chunk that does not contain the fact *and its label* is a Phase 2 bug, not a Phase 5 bug.
7. **Size check.** Confirm no chunk exceeds 150 words and that overlap appears only between prose chunks, never inside a fact group.
8. **Tests.** `pytest` passes on `test_extraction.py` and `test_chunking.py`.
9. **Robustness.** Temporarily blank one cached HTML file and re-run — it must fail loudly and name the URL, not ingest an empty page.

**Do not move on until:** steps 3, 4, 5, and 6 all pass. These are the checks that make the difference between a bot that is compliant and one that is not. Step 3 in particular is a hard gate.

**Watch for:** two failure modes account for most of the pain here. (a) Selecting sections by CSS class instead of heading text — Groww's classes are content-hashed, e.g. `exitLoadStampDutyTax_popupBody__4ywku`, and break without warning. (b) Broadening the allowlist to "get more facts in" — resist it; a missing fact is a graceful "not in my sources", a leaked return is a compliance breach.

---

## Phase 3 — Embedding & vector store

**Goal:** a persisted 384-dim Chroma store built once, from the chunks already proven safe in Phase 2.

### Files to create

| File | Purpose |
|---|---|
| `rag/embedder.py` | Single `SentenceTransformer` instance, `all-MiniLM-L6-v2`, CPU, normalised embeddings. Lazy-loaded so importing the module is cheap |
| `rag/store.py` | `PersistentClient`, collection `hdfc_faqs`, cosine space, 384-dim. Upsert keyed on `content_hash`; also writes `data/embeddings_preview.txt` |
| Extend `ingest.py` | Wire the embed and store stages; the load-and-chunk mode from Phase 2 becomes the default full run |
| `tests/test_store.py` | Dimension, count, persistence, idempotency, and encoder-mismatch assertions |
| `tests/test_embedder.py` | Model-cache detection and the dimension guard, without needing the weights |

### What this phase does

- Loads MiniLM once and reuses that exact instance for both chunks and, later, questions. Two differently-configured encoders make cosine distance meaningless, so this is a correctness requirement (C7), not a performance one.
- Embeds each chunk to a 384-dim vector and writes it to a persistent Chroma collection on disk, along with the full metadata set from `CHUNKING.md` §3.
- Makes ingestion idempotent: re-running with unchanged `content_hash` is a no-op, so an accidental double-run never doubles the corpus.
- Guarantees the store survives a restart — ingestion is a separate command from serving, and the app never re-embeds.

### How to verify

1. **Run the full pipeline.** `python ingest.py`. Expect one model load, then embedding progress, then a chunk count equal to Phase 2's count — **the numbers must match exactly**. A mismatch means records were dropped between phases.
2. **Dimension check.** Query the collection directly and confirm every vector is 384-dimensional, and that the count equals the number of chunks in `chunks.txt`.
3. **Persistence (this is F8).** Quit Python entirely, start a fresh process, and confirm the collection is readable with the same count. Re-run `ingest.py` and confirm it reports the corpus is already up to date and changes nothing.
4. **Metadata check.** Pull one record back and confirm all ten metadata fields are present and correct — `source_url`, `scheme_name`, `scheme_code`, `category`, `section`, `fields`, `chunk_index`, `retrieved_at`, `content_hash`.
5. **Sanity-search.** Run a trivial cosine search by hand (not via the retriever, which does not exist yet) for the text "lock-in period ELSS" and confirm the top result is the ELSS lock-in chunk, and that a search for "exit load Small Cap" returns the Small Cap exit-load chunk.
6. **Distance sanity.** Note the raw distances returned for a good match and for a nonsense query. These two numbers are the raw material for the Phase 5 threshold, so write them down now.
7. **Rebuild test.** Delete `data/chroma/` and re-run to confirm a clean rebuild works from the cached HTML with no network.
8. **Commit check.** `git status` — `data/chroma/` must be ignored; `data/chunks/chunks.txt` and `data/embeddings_preview.txt` must be tracked.
9. **Embeddings preview.** `data/embeddings_preview.txt` dumps the first 5 stored vectors, first 10 dimensions each, with the scheme, section, `content_hash`, true dimensionality, and L2 norm. The values are read **back out of Chroma**, not reused from the in-flight batch, so the file is evidence of what is on disk. Every vector must show `384` dimensions and an L2 norm of `1.000000` — a norm of anything else means the encoder was not normalised, which would break the Phase 5 threshold.

### Measured results (2026-09-29)

| Check | Result |
|---|---|
| Chunk count | 31 stored = 31 in `chunks.txt` (exact match) |
| Dimensions | 384 on every vector |
| Metadata | all 9 fields present and non-empty on all 31 records |
| Persistence | reopened in a fresh process, count unchanged |
| Idempotency | second run reported `0 added, 31 unchanged` |
| Clean rebuild | deleted `data/chroma/`, rebuilt from cache with the network blocked |
| Store count vs. citations | all `source_url`s and `scheme_code`s inside the allowlist |
| Persistence across restart | corpus sha256 `16585b7b39f2a232ec5d2fec4cdbb9c2` identical in two separate processes — byte-for-byte, not just the same count |
| Embeddings preview | `data/embeddings_preview.txt`, 5 vectors x 10 dims, all 384-dim, all L2 norm `1.000000` |

### Retrieval measurements — raw material for the Phase 5 threshold

Cosine distance is lower-is-better. Measured over the 8 in-scope questions from PRD §4.1–4.2, taking the distance to the chunk that actually holds the answer:

| | min | median | max |
|---|---|---|---|
| Distance to the **correct** chunk | 0.138 | 0.160 | 0.374 |
| Distance to **off-topic** top-1 | 0.823 | 0.928 | 0.945 |

The separation is wide and the configured `DISTANCE_THRESHOLD = 0.45` sits inside the gap, rejecting **0 of 8** correct answers. Keep it. One caveat: terse keyword queries run worse than natural questions — `"lock-in period ELSS"` scores 0.574, which 0.45 would reject. Phase 5 will receive natural-language questions, not keywords, so this is a note rather than a change.

### Two findings that constrain Phase 5

**1. Cosine alone misranks across schemes. A lexical scheme filter is required, not optional.**
Ranking all 31 chunks by pure cosine, the chunk holding the correct answer came **first for only 6 of 8** questions. The two failures were cross-scheme confusion between near-duplicate chunks — "exit load on HDFC Small Cap Fund" put the *Large Cap* exit-load chunk first, because both chunks are mostly exit-load text and MiniLM is weak on the keyword "Small".

Adding a scheme-name/category match before ranking takes this to **8/8**. This is not an optimisation; a bot that answers a question about one fund with another's exit load is worse than one that says "not in my sources". Phase 5 must filter on scheme before ranking.

**2. Do not rely on top-1. The required facts are sometimes at rank 5.**
"What risk category does HDFC Balanced Advantage Fund fall under?" puts the right chunk at **rank 5 of 6** for that scheme, with a margin of one position. The chunk that ranks *first* is the About prose, which does not contain the rating — the fact lives in the `overview` chunk. `top_k = 5` happens to include it, but only just.

The cause is a Phase 2 exclusion decision: the About prose sentence "…is rated Very High risk" was dropped as duplicative of the overview fact, which is correct for avoiding two conflicting numbers, but it left the About chunk semantically "about the fund" without carrying the risk answer. Two options for Phase 5, in order of preference: keep `top_k = 5` *and* have the answer step check whether any chunk in the window actually covers the question's field before saying "not in my sources"; or, if a query is still unanswered, widen the window. Re-chunking to fix this would mean invalidating the Phase 2 artifact and re-embedding the corpus, which is not worth it for a question that already resolves.

**Do not move on until:** the store survives a restart, the chunk count matches, and every vector is 384-dim.

**Watch for:** the first model download is ~90 MB and needs network; subsequent runs are offline. That is now enforced rather than assumed — `rag/embedder.py` checks the HuggingFace cache directory for weights and passes `local_files_only=True` when they are there. Without it, sentence-transformers issues a HEAD request on every load to check for a newer revision, and a network-blocked run spent **145s** in retry backoff for a model it already had. It now takes **5.9s**. If the store is committed accidentally, every later phase is testing stale data — check `.gitignore` now rather than debugging a phantom dimension mismatch later.

---

## Phase 4 — Guardrails

**Goal:** make the bot refuse advice and PII questions before any LLM call, and be able to check its own output afterwards. Pure logic, fully testable with no API key.

### Files to create

| File | Purpose |
|---|---|
| `rag/guardrails.py` | `check_pii()`, `is_advice()`, `validate_output()` |
| `rag/prompts.py` | The system prompt and the answer contract, versioned in one place. Contains no secrets |
| `tests/test_guardrails.py` | The 6 refusal cases and 3 PII cases from PRD §4.3 and §4.4 |
| `tests/test_eval_set.py` | Runs the eval query list through the guardrails only — no retrieval, no LLM |

### What this phase does

- **PII detection** for PAN, Aadhaar, account numbers, OTPs, email addresses and phone numbers. On a hit, it returns a refusal and the matched value is never echoed back and never logged (C5).
- **Advice classification** for buy/sell/hold, "which is better", "is now a good time", and return-chasing questions. On a hit, it returns a polite facts-only refusal plus a relevant *educational* link.
- **Output validation** for answers the LLM has already produced: sentence count, exactly one citation, the citation must be one of the 5 allowlisted URLs, and regex checks for advice or performance language. A violation is replaced with a safe fallback rather than shown.
- Keeps the refusal text, the educational links, and the answer contract in `prompts.py` so the wording used by the UI, the README, and the sample Q&A file cannot drift apart.

### How to verify

1. **PII cases.** Run the 3 PRD §4.4 inputs. Each must be refused. Then grep the test output and any logs for `ABCDE1234F`, `12345678901`, and the sample email — **expect zero hits anywhere**, including logs (F6).
2. **Advice cases.** Run the 6 PRD §4.3 questions. Each must be refused with a link, and **none** may contain a recommendation, a scheme name being recommended, or a hedge that implies advice (F5).
3. **False-positive check.** Run the 3 welcome-screen questions and the 7 in-scope questions from PRD §4.1–4.2 through the classifier. **All 10 must pass through as factual.** This is the check that prevents an over-eager guardrail from refusing "What is the expense ratio?" — easy to do with a naive keyword list containing "best" or "should".
4. **Refusals are free.** Confirm the refusal path returns without calling the LLM — assert no API call is made. Refusals must stay fast and quota-free.
5. **Output validation.** Feed `validate_output()` five crafted answers: one with 5 sentences, one with two links, one citing a non-allowlisted URL, one containing "annualised returns of 12%", one clean. Expect four rejections and one pass.
6. **PII false positives.** "What is my expense ratio?" and "What is the minimum SIP?" contain no PII. Both must pass. Watch specifically for the account-number regex catching a long digit string in a legitimate question.
7. **Tests.** `pytest tests/test_guardrails.py` green; `pytest` overall green.

### Measured results (2026-09-29)

| Check | Result |
|---|---|
| Advice refusals (PRD §4.3) | 6/6 refused, all with a link, none containing a recommendation |
| PII refusals (PRD §4.4) | 3/3 refused, plus Aadhaar, phone and OTP cases |
| Factual false positives | **10/10** in-scope questions pass through |
| Off topic | 4/4 refused with "I don't know" |
| Output validation | 4 crafted answers rejected (too long / two links / non-allowlisted URL / performance language), 1 clean answer passed |
| PII false positives | "What is my expense ratio?", "₹500 minimum", "1% within 1 year" all clean |
| Refusals make no network call | asserted at the socket layer; guardrails import no client library |
| PII in the repo | zero hits outside `docs/`, including `.pytest_cache` |

### Three things found while building this

**1. pytest was writing the PII into its own cache.** The tests that exist to prove PII is never stored were themselves storing it: `@pytest.mark.parametrize` derives each test's node ID from the parametrised string, and those node IDs are written to `.pytest_cache/v/cache/nodeids`. Grepping the repo found the PAN, the account number, and the email sitting in the pytest cache. Fixed by giving the PII cases opaque `ids=("pan", "account", ...)`. This is precisely the failure mode Phase 4 warns about — a PII value in a log line is an automatic §5.4 failure "even if the UI behaves correctly" — except the log was the test harness.

**2. The PII fixtures are assembled from string fragments on purpose.** `PAN = "ABCDE" + "1234" + "F"` keeps the literal out of every tracked file while still exercising the real regex. An earlier version of `test_eval_set.py` asserted `"ABCDE1234F" not in text`, which satisfied the letter of the rule while planting the value in a source file.

The remaining occurrences live in `docs/PRD.md` and this file, where they appear as the *test cases to grep for*. They are synthetic examples in a requirements document, not runtime data, and the plan scopes the grep to "the test output and any logs". Redacting them would make the specification less precise; if the evaluator greps the whole repo instead, they are the only two hits and can be masked without loss.

**3. The "Last updated" footer was silently eating a sentence.** The contract allows 3 sentences, but the footer and the citation are appended to the answer and were being counted, so a compliant 3-sentence answer failed as "4 sentences". `answer_body()` now strips the footer and any URL before counting, so the budget is 3 sentences of prose as written. Worth noting because the same trap hides in the sentence counter itself: `1.03%` and `₹1,426.93` must not split on the decimal point, so the split requires terminal punctuation *followed by whitespace*.

### One PRD assumption that does not hold

PRD §4.2 lists "How do I download my capital gains statement?" as an in-scope question, but **no chunk in the corpus mentions a statement** — Phase 2's allowlist expected a statement-guide section that these pages do not expose. Rather than refuse it, the scope check treats it as in-domain and lets it through, and Phase 5 will answer it with "I don't know" plus a link. That is the honest outcome: a gap in the corpus is not a bad question. Worth either adding a source or dropping the question from §4.2 before the demo.

**Do not move on until:** 6/6 refusals, 3/3 PII, **10/10 factual questions pass through**, and the PII values appear nowhere in output or logs.

**Watch for:** the two failure modes that matter are (a) a guardrail that refuses legitimate factual questions — a bot that only says no is not a demo, and (b) PII values leaking into a log line, which is an automatic PRD §5.4 failure even if the UI behaves correctly. Check the logs explicitly rather than assuming.

---

## Phase 5 — Retrieval and LLM answer

**Goal:** the full query path — question to grounded, cited, three-sentence answer. First phase that needs the Groq key.

### Files to create

| File | Purpose |
|---|---|
| `rag/retriever.py` | Embed the question with the same MiniLM instance; `k=5` cosine search; optional `scheme` metadata filter; distance threshold |
| `rag/generator.py` | Builds the grounded prompt from retrieved chunks; calls Groq at `temperature=0`, `max_tokens=1024`; attaches citation and `Last updated from sources`; one retry on transient API error |
| `config.py` (update) | Confirm the live free-tier Groq model id; set `GROQ_API_KEY` in `.env` |
| `tests/test_retriever.py` | Threshold behaviour, scheme filtering, out-of-corpus handling |
| `tests/test_generation.py` | Answer contract assertions, run against a small subset to limit free-tier quota use |
| `eval/queries.txt` | The 10-query evaluation set, including the out-of-corpus and refusal probes |

### What this phase does

- Detects the scheme named in the question and applies it as a Chroma metadata filter, so a question about HDFC ELSS can never retrieve a Small Cap chunk even if one scores marginally higher.
- Compares the best distance against a threshold. Below it, the bot says the answer is not in its sources and links out — it never guesses (F10).
- Assembles the retrieved chunks into a prompt that instructs the model to answer **only** from the given context, to stay within three sentences, to give no recommendation, and to state when the context does not contain the answer.
- Calls Groq deterministically at `temperature=0` with a tight token budget, then attaches exactly one citation from the allowlist and the `Last updated from sources` line.
- Runs the result through `validate_output()` before it leaves the function, so a contract violation becomes a safe fallback rather than a bad answer on screen.

### How to verify

1. **Key and model.** Confirm `GROQ_API_KEY` is in `.env`, git-ignored, and that the configured model id is actually available on the free tier — availability rotates, so this fails in a way that looks like a code bug.
2. **The 3 welcome questions.** Ask all three and check each against the source page. Expect: Large Cap expense ratio `1.03%`; ELSS lock-in `3 years`; Balanced Advantage minimum SIP `₹100`. (These match the ground-truth table above, re-read from the live pages; an earlier draft of this step said `0.78%` and `₹500`, which the pages contradict.)
3. **The 7 in-scope questions** (PRD §4.2). Each must return a correct fact, ≤3 sentences, exactly one allowlisted link, and the `Last updated from sources` line. **Target ≥8/10 clean** (Q1, Q2).
4. **Retrieval relevance.** For each query, log the top-5 chunk sections. The correct chunk must appear in the top 5 for at least 8 of 10 queries. If a query misses, inspect whether the scheme filter or the threshold is responsible before touching the prompt.
5. **Out-of-corpus.** Ask "What is the expense ratio of HDFC Mid-Cap?" — a scheme not in the corpus. Expect "not in my sources" plus a link, and **no number at all** (F10).
6. **Refusals still work end to end.** Run the 6 advice questions and 3 PII questions through the full path. Expect refusal, no LLM call, no PII in logs.
7. **Threshold calibration.** You noted two raw distances in Phase 3 step 6. Confirm the configured `DISTANCE_THRESHOLD` sits between "good match" and "nonsense query", then **freeze it and write the number into the README**. An undocumented threshold cannot be defended in the demo.
8. **Contract assertions.** Script-check every answer in the sample set for ≤3 sentences and exactly one allowlisted link. Zero violations expected (F2, F3, F4).
9. **No performance claims.** Grep the full answer set for `annualised`, `CAGR`, `rank`, `category average`, `best performing`, `% returns`. Expect zero hits (F7).
10. **Latency.** Time a cold run. Target <8 s end to end (Q3).
11. **API failure.** Temporarily set a wrong key. Expect a clear error, never a fabricated answer.

**Do not move on until:** steps 2, 5, 6, 8 and 9 all pass with zero violations. Steps 1 and 3 may be slightly imperfect (free-tier quota) but must be substantially correct.

### Measured results (2026-09-29)

**Model.** Live key in git-ignored `.env`; `GROQ_MODEL=openai/gpt-oss-120b` confirmed present in the free-tier model list at verify time.

| Check | Result |
|---|---|
| Welcome Q1 expense ratio | Large Cap `1.03%`, correct |
| Welcome Q2 lock-in | ELSS `3 years`, correct |
| Welcome Q3 min SIP | Balanced Advantage `₹100`, correct |
| In-scope eval set | **9/10** clean grounded answers + 1 honest "I don't know" (capital-gains statement, PRD §4.2 — no chunk mentions one) |
| Answer contract | every answer ≤2 sentences, exactly one allowlisted link, `Last updated from sources` line present |
| Retrieval relevance (step 4) | correct chunk in top-5 for every answerable query; risk question's answer at rank 5 as documented |
| Scheme filter | out-of-corpus probes (Mid-Cap, Parag Flexi Cap, Nifty 50 Index) returned `not_in_sources` with **no number** (F10) |
| Refusals end-to-end (step 6) | 6/6 advice + 4/4 off-topic refused with **no LLM call**; PAN not echoed |
| Threshold (step 7) | frozen `DISTANCE_THRESHOLD = 0.45` (config) — good matches 0.138–0.374, off-topic 0.823+ |
| Contract assertions (step 8) | 0 violations across the sample set |
| No performance claims (step 9) | zero `annualised` / `CAGR` / `rank` / `best performing` in the answer set |
| Latency (step 10) | cold end-to-end run **0.9 s** (target <8 s) |
| API failure (step 11) | wrong key → `GenerationError: Groq API error 401`, no fabricated answer |

**CLI.** `python ask.py` prompts interactively and prints, for every question, the scheme filter, the retrieved chunks with their cosine distances and fields, and the grounded answer; `--retrieval-only` exercises retrieval without any API key; `python ask.py "question"` is the one-shot form.

**Unchanged by design:** the threshold number lives only in `config.py`; the test suite never calls the real API (injected fake client) except an opt-in live test behind `RUN_LIVE_GROQ=1`.

**Watch for:** three likely problems, in order. (a) The model volunteering a return figure from its own memory even when the corpus excludes it — this is why the prompt says to state when the context lacks the answer. (b) The threshold being set by feel rather than from the Phase 3 numbers. (c) The free tier running out of quota mid-testing, which looks like a code failure — run the eval set in small batches.

### Follow-up question rewriting (added after the Phase 5 review)

A follow-up like "what about its fees?" retrieves nothing on its own, because
"its" carries no scheme name. `rag/conversation.py` keeps the last
`CONVERSATION_MESSAGES` (default 10) messages in RAM and rewrites a follow-up
into a standalone question before retrieval.

The rewriter is prompted to resolve references and *nothing else* — it may not
answer, add facts, or explain itself — so a rewrite can never introduce a fact
that was not already retrieved and cited. A lexical gate (`ANAPHORA` /
`FOLLOWUP_OPENER`) decides whether a question is worth an LLM call at all, so
self-contained questions and the first turn of a session cost nothing.

Ordering is the load-bearing part. The guardrail runs on the **original**
question before any rewrite, so a PAN never reaches the rewriter and that turn
is never stored. One rule is relaxed on purpose: `check_question` requires a
question to name a scheme, which a follow-up never does, so an **off-topic**
refusal is the only one a rewrite may rescue — and only after the resolved
question passes the full guardrail again. PII, advice, and cross-scheme
refusals are never rescued. Both `ask()` and the CLI go through
`resolve_question()` so the two paths cannot disagree.

Measured against the live model (`openai/gpt-oss-120b`):

| Check | Result |
|---|---|
| Pronoun follow-ups | "what about its exit load?" → "What is the exit load for HDFC ELSS Tax Saver Fund?"; answered `nil` (correct) |
| Chained follow-ups | "and its minimum SIP amount?" on the same turn → `₹500` (correct) |
| Scheme switch | after Large Cap, "what about its exit load?" resolved to Large Cap, not the previous scheme |
| Mid-conversation switch | "actually what about HDFC Small Cap Fund?" then "and its exit load?" → Small Cap, not Large Cap |
| Off-topic rescue | "what about its risk rating?" refused alone, answered `Very High` for Balanced Advantage after resolution |
| PII follow-up | refused as `pii` with **no rewriter call**, **no LLM call**, and the turn absent from `history` |
| Advice / cross-scheme | refused with no rewriter call |
| Unresolvable referent | after "what is the weather today?", "what about its risk rating?" stayed refused — the rewriter declined to invent a referent |
| Rewrite failure | degrades to the original question and the honest "not in sources" answer; no crash, no fabrication |
| Window bounds | buffer capped at 10 messages, oldest dropped first; `reset` clears it |

### Bug found and fixed while verifying this

`validate_output`'s `unfinished_sentence` check compared the answer body's last
character to `.`/`!`/`?`. The model legitimately writes answers ending in a
closing mark ("...within the first year.)"), which were rejected as truncated
and fell back to "I don't know". The check now strips trailing closing
brackets and quotes first. Two causes were in play here: this false positive,
**and** a genuine one — see the note in "Known limits" on `max_tokens`.

---

## Phase 6 — User interface

**Goal:** the demo surface — welcome line, 3 example questions, chat thread with a citation under every answer, persistent disclaimer, and a loading state.

### Deviation: Streamlit, not FastAPI

This section originally specified FastAPI + a hand-written `static/index.html` +
`render.yaml`. It was built as a **Streamlit** app instead, which PRD §7 already
allowed ("`uvicorn main:app` **or** `streamlit run app.py`"). The reason is that
the hand-written UI would have re-implemented rendering that `prompts.py` and
`config.py` already own — the disclaimer, the refusal wording, the
`Last updated` line — and every one of those would need re-testing in two
places. Streamlit draws what the existing modules produce, so the compliance
behaviour has exactly one implementation.

Not built, and still open: `render.yaml` and the Render deploy (step 11), the
`/api/ask` and `/health` endpoints, and `docs/sources.md` /
`docs/sample_qa.md` from the deliverables checklist.

### Files to create

| File | Purpose |
|---|---|
| `app.py` | Streamlit entry point. Draws the thread, the Sources expander, the clear-chat button. No answering logic |
| `rag/chatsession.py` | The view model: `BotTurn`, `SourceRow`, `ChatSession`, `build_turn`, `startup_problems`. Pure, no Streamlit import, so it is unit-testable |
| `tests/test_ui.py` | The view model: sources, refusals, verbatim text, thread vs. window bounds, startup checks |
| `tests/test_ui_app.py` | Headless `AppTest` smoke tests that the script renders, and that a missing store is a sentence rather than a traceback |
| `render.yaml` | **Not done.** `startCommand` would be `streamlit run app.py --server.port $PORT`, `buildCommand` installs + `ingest.py` |
| `docs/sources.md` | The 5 URLs (P3) — **not done** |
| `docs/sample_qa.md` | 5–10 real queries with the assistant's actual answers and links (P4) — **not done** |

### What this phase does

- Renders a chat thread with the 3 example questions as one-click buttons, the welcome line on an empty session, and a permanent **"Facts-only. No investment advice."** disclaimer.
- Puts a **Sources** expander under every answer: the exact chunks it was grounded in, with scheme, section, cosine distance, the fields covered, and the chunk text. The distances shown are the ones `DISTANCE_THRESHOLD` was calibrated against.
- Renders the three refusal types through one `st.info` panel with the educational link, and **shows no expander** — a refusal is produced before retrieval, so there is nothing behind it.
- Shows a spinner during retrieval and generation, because a free-tier cold start reads as a broken page during a live demo.
- Surfaces the resolved follow-up when a question was rewritten, so a pronoun is never a black box.
- Keeps a **Clear chat** button that empties both the thread and the memory window.
- Warms the encoder once per server process via `st.cache_resource`, so the first question does not pay the MiniLM load.

### Deliberate boundaries

- **The answer text is never reformatted.** The disclaimer, the `Last updated from sources` line and the single allowlisted citation are part of the tested answer contract; re-rendering them here would break the contract the tests assert.
- **The thread is unbounded, the memory window is not.** A user should see every turn they asked for, while `Conversation` stays capped at `CONVERSATION_MESSAGES` and in RAM.
- **A PII-refused turn is shown but not remembered.** The user sees the refusal they earned; the identifier is not kept in the window.
- **No answer is ever computed in the UI.** `app.py` calls `ChatSession.ask`, which is the same `rag.conversation.ask` the CLI uses, so the guardrail ordering is identical on both paths.

### How to verify

1. **Start locally.** `streamlit run app.py`. Open the printed URL.
2. **First impression.** Welcome line, 3 example questions, and the disclaimer are visible without scrolling. Clicking an example submits it.
3. **All 3 welcome questions** return correct answers with a visible citation, the `Last updated` line, and a Sources expander.
4. **Refusal rendering.** Ask one advice question, one PII question, and one off-topic question. All three show the refusal panel with a link and **no** expander. The PII value is not echoed anywhere in the assistant's output.
5. **Out-of-corpus rendering.** Ask about HDFC Mid-Cap. Expect the not-in-sources panel with a link and no expander.
6. **Loading state.** Submit a question and confirm the spinner appears immediately and clears on response.
7. **Follow-ups.** Ask about ELSS lock-in, then "what about its exit load?" — expect the resolved question shown and the answer grounded in ELSS, not the previously discussed scheme.
8. **Clear chat.** Press it. The thread empties and the welcome line returns.
9. **Statelessness (C12).** Ask a question, hard-refresh, and confirm the thread is empty.
10. **Disclaimer exactness.** Compare the UI string against the README and `config.DISCLAIMER` **character for character** (P5).
11. **Missing setup.** With `data/chroma/` deleted, confirm the page shows a sentence telling you to run `ingest.py` rather than a traceback.

### Measured results (2026-09-29)

Verified with Streamlit's headless `AppTest`, which runs the real script.

| Check | Result |
|---|---|
| App renders | no exception; title, disclaimer, chat input, 4 buttons present |
| Welcome Q1 via UI | Large Cap `1.03%`, correct; Sources expander shows 5 chunks, best = `overview` at d=0.216 — the chunk that holds the figure |
| Follow-up via UI | "what about its exit load?" → shown as "What is the exit load for HDFC ELSS TaxSaver Fund?", answered `nil` (correct) |
| Advice / off-topic / PII | 3/3 rendered as refusal panels with a link, **0 expanders** |
| PII not echoed | PAN absent from every assistant bubble, caption and expander |
| Clear chat | 4 messages → 0, welcome line restored |
| Missing store | rendered as an `st.error` naming `ingest.py`, no exception |
| LLM calls in tests | none — every smoke test runs the app to its resting state |
| Suite | `307 passed, 1 skipped` before the app smoke tests, `314 passed, 1 skipped` after; the app tests add ~2 s because the encoder is already loaded |

### Not verified (needs a browser or a deploy)

Render deploy, the demo video, and the `/health` endpoint — the last of these
only if the FastAPI route comes back.

**Watch for:** (a) The disclaimer drifting from the README — it is a graded deliverable and word-for-word comparison is the check. Here it is rendered from `config.DISCLAIMER`, so the risk is a hand-typed copy somewhere else, not this file. (b) `st.cache_resource` holding the encoder across a re-ingest, so a `data/chroma/` rebuild in a running session can serve stale vectors; restart the app after re-ingesting. (c) Free-tier cold start, which the spinner now covers but which is still slow on the first request of a fresh process.

---

## Final pass — deliverables checklist

Tracked separately from the six phases, since it is what gets submitted.

- [ ] `docs/sources.md` — the 5 URLs
- [ ] `docs/sample_qa.md` — 5–10 queries, real answers, links
- [ ] `README.md` — setup, scope, known limits, frozen retrieval parameters, disclaimer snippet
- [ ] Disclaimer matches UI character for character
- [ ] `data/chunks/chunks.txt` committed and readable
- [ ] `.env` ignored; no key in any committed file
- [ ] `requirements.txt` pinned
- [ ] Full `pytest` run green
- [ ] ≤3-min demo video recorded
- [ ] Live URL on Render, or video only

---

## Two things worth knowing before you start

**The compliance checks are the milestone, not the chat.** Phases 2 and 4 hold every constraint that can actually be violated — a leaked return figure, a cross-scheme answer with a correct-looking citation, a PAN in a log line. The LLM layer is comparatively forgiving; a `temperature=0` call with three good chunks and a clear contract will behave. Do not let the UI work pull you into skipping the Phase 2 and Phase 4 verification steps, because those are the checks you will be asked to demonstrate live.

**Commit `data/chunks/chunks.txt` early and re-read it after every extraction change.** It is a graded deliverable, it is the fastest way to see whether a rule change broke something, and it is the artefact that makes the whole pipeline explainable without running a single line of code.
