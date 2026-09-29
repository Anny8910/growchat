# PRD — Mutual Fund Facts-Only RAG Chatbot (HDFC AMC)

**Version:** 1.0
**Date:** 2026-09-29
**Source:** `docs/problemstatement.txt`
**Status:** Draft for class milestone review

---

## 1. Goal

Build a small, working Retrieval-Augmented Generation (RAG) chatbot that answers **factual** questions about a scoped set of HDFC mutual fund schemes using **only** a fixed corpus of public web pages, and cites a source link in every answer.

The bot exists to demonstrate a complete, inspectable RAG pipeline — **Ingestion (Load → Chunk → Embed → Store)** and **Query (Question → Embed → Retrieve → LLM → Answer)** — end to end, with zero investment advice.

**Success looks like:** a user asks "What's the exit load on HDFC Large Cap?" and within a few seconds receives a ≤3-sentence factual answer, one clickable source link, and a "Last updated from sources" line — with no hallucinated numbers.

### Non-goal

This is a **facts retrieval tool**, not a financial advisor, recommendation engine, or return-comparison product.

---

## 2. Target Users

| User | Need | What success means to them |
|---|---|---|
| **Retail investor** comparing HDFC schemes | Quick, trustworthy factual details (expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark) without digging through PDFs | Gets the exact number plus a link they can verify themselves |
| **Support / content team** answering repetitive MF questions | A reusable first-line answer to the ~80% of questions that are pure facts | Cuts repeat lookups; escalates the rest to a human |
| **Evaluator / instructor** (class milestone) | Proof the RAG pipeline is real, not a wrapper on a chatbot | Sees the chunk file, the vector store, the retrieval step, and citations in the UI |

### Core user need (in their words)

> "I want to know one fact about one HDFC scheme, fast, and I want to see exactly where that fact came from."

---

## 3. Scope

### 3.1 Corpus — AMC and schemes

**AMC:** HDFC Asset Management (single AMC, as required).

**Schemes (5, all Direct – Growth):**

| # | Category | Scheme | Source URL |
|---|---|---|---|
| 1 | Large Cap | HDFC Large Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| 2 | Flexi Cap | HDFC Equity (Flexi Cap) Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| 3 | ELSS | HDFC ELSS Tax Saver Fund – Direct Plan – Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| 4 | Small Cap | HDFC Small Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| 5 | Balanced Advantage (Hybrid) | HDFC Balanced Advantage Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

**Corpus rule:** exactly these 5 public pages. No third-party blogs, no scraped app back-ends, no screenshots. The URL list ships as `docs/sources.md` (deliverable).

### 3.2 In scope

**A. Facts-only Q&A**
- Expense ratio / TER
- Exit load (incl. tiered slabs, where stated)
- Minimum SIP amount and minimum lump sum
- ELSS lock-in period and Section 80C treatment (as stated in source)
- Riskometer category and benchmark
- Minimum investment / NAV-related published facts
- "How do I download my capital-gains / account statement?" (guidance steps only)

**B. RAG pipeline (both stages, per brief)**
- **Ingestion:** Load → Chunk → Embed → Store
  - Load the 5 URLs, clean HTML to text, retain per-page structure
  - Agent inspects the data and proposes a justified chunking strategy **before** code is written → see `docs/CHUNKING.md`
  - All chunks dumped to a human-readable `chunks.txt` for inspection
  - Embed with `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local, no API key)
  - Persist to **ChromaDB on disk** — ingestion runs once, not on every restart
- **Retrieval:** Question → embed with the **same** MiniLM model → retrieve top-k chunks → LLM (Groq) → answer

**C. Answer contract (hard rules enforced in the prompt + validated in code)**
- ≤ 3 sentences
- Exactly **one** clear citation link per answer
- Includes a `Last updated from sources: <date>` line
- If retrieval confidence is below threshold, say it doesn't have the answer and point to the source page — never fill the gap from model memory

**D. Refusal behaviour**
- Politely refuses opinionated, predictive, or portfolio questions ("Should I buy/sell X?", "Which is better?", "Is now a good time to invest?")
- Refusal message includes a relevant **educational** link, not advice
- Also refuses any question containing PII (PAN, Aadhaar, account number, OTP, email, phone) without echoing the value back

**E. Tiny UI**
- Welcome line
- 3 clickable example questions
- Persistent disclaimer: **"Facts-only. No investment advice."**
- Chat thread with a visible source link under each answer
- Loading / thinking state so the demo doesn't look frozen

**F. Deliverables (per brief)**
1. Working prototype (app), plus a ≤3-min demo video as backup
2. Source list (MD) of the 5 URLs
3. README: setup steps, scope (AMC + schemes), known limits
4. Sample Q&A file (5–10 queries with answers + links)
5. Disclaimer snippet as used in the UI
6. This PRD

**G. Repo hygiene**
- `.env` for the Groq key, `.gitignore`d, never committed
- `requirements.txt` pinned
- `README.md` with setup steps

### 3.3 Out of scope

| Excluded | Why |
|---|---|
| Investment advice, buy/sell/hold calls, suitability judgement | Explicit constraint — facts-only |
| Return, CAGR, or performance comparison/ranking across schemes | Explicit constraint — link to the official factsheet instead |
| Fund recommendation, portfolio allocation, asset allocation advice | Explicit constraint |
| Schemes outside the 5 listed; other AMCs | Corpus is scoped to one AMC |
| Buying/selling/transacting, login, order placement, account management | Read-only information product |
| User accounts, sign-up, chat history persistence, analytics, telemetry | Not needed; also avoids PII storage |
| Uploading user documents (statements, PAN, KYC) | No PII handling, by design |
| Live NAV, real-time prices, market data feeds | Source pages are static facts; no live data source available |
| Voice/multilingual input, mobile app, bulk document ingestion | Out of milestone; keep scope small |
| OCR / PDF factsheet parsing | Corpus is the 5 web pages |
| Automated re-crawling / scheduled refresh | Static snapshot is acceptable for the demo; manual re-ingest command suffices |

---

## 4. Example User Questions

### 4.1 The 3 welcome-screen questions (one per category, must work flawlessly)

1. **What is the expense ratio of the HDFC Large Cap Fund?**
2. **What is the lock-in period for HDFC ELSS Tax Saver Fund?**
3. **What is the minimum SIP amount for HDFC Balanced Advantage Fund?**

### 4.2 In-scope factual questions

- "What is the exit load on HDFC Small Cap Fund?"
- "What's the benchmark of HDFC Equity (Flexi Cap) Fund?"
- "What risk category does HDFC Balanced Advantage Fund fall under?"
- "Can I redeem HDFC ELSS before 3 years?"
- "What is the minimum lump sum investment for HDFC Large Cap?"
- "How do I download my capital gains statement?"
- "What is the NAV of HDFC Flexi Cap?" *(only if the page shows a NAV; otherwise the bot must say the page doesn't list it and link out)*

### 4.3 Must-refuse questions

- "Should I invest in HDFC ELSS or HDFC Flexi Cap?"
- "Is now a good time to buy the HDFC Large Cap Fund?"
- "Which HDFC fund has given the best returns?"
- "If I invest ₹10,000/month for 10 years, what will I get?"
- "Is HDFC Small Cap safe for my retirement?"
- "Tell me if I should sell my HDFC Flexi Cap."

### 4.4 PII-must-not-store examples

- "My PAN is ABCDE1234F, can you check my ELSS eligibility?"
- "My account number is 12345678901 — what's my exit load?"
- "Share my registered email for the statement download link."

---

## 5. Success Criteria

Each criterion is binary and demo-verifiable.

### 5.1 Functional (must pass all)

| # | Criterion | Verification |
|---|---|---|
| F1 | All 3 welcome-screen questions return a correct factual answer with ≥90% numeric accuracy vs. the source page | Manual check against source |
| F2 | **100%** of factual answers contain exactly one working source link from the 5-URL allowlist | Inspect every answer in the sample Q&A file |
| F3 | **100%** of factual answers are ≤ 3 sentences | Sentence-count check |
| F4 | **100%** of answers include a "Last updated from sources" line | Inspect |
| F5 | All 6 must-refuse questions return the polite facts-only refusal + educational link, and **never** a recommendation | Scripted test pass |
| F6 | PII inputs are refused and the PII value is not echoed or stored | Scripted test pass; logs inspected |
| F7 | No answer contains a return, CAGR, ranking, or comparison claim | Grep sample answers for performance language |
| F8 | Ingestion runs once; a second app start uses the persisted Chroma store with no re-embed | Delete nothing, restart, observe no re-ingest log |
| F9 | `chunks.txt` is committed and human-readable; every chunk shows source URL + metadata | Open the file |
| F10 | Out-of-corpus question ("What is the expense ratio of HDFC Mid-Cap?") is answered with "not in my sources" + link, not a guess | Manual test |

### 5.2 Quality (the graded bar)

| # | Criterion | Target |
|---|---|---|
| Q1 | Answer groundedness — every factual claim traceable to a retrieved chunk | Human review of 10-query sample: ≥ 9/10 clean |
| Q2 | Retrieval relevance — correct chunk in top-k for in-scope questions | Manual check: ≥ 8/10 |
| Q3 | Answer latency | < 8 s end to end on a cold local run |
| Q4 | Demo robustness — 5 consecutive scripted questions with no crash | 5/5 pass |
| Q5 | The pipeline is explainable in the demo: evaluator can point to loader, chunker, embedder, store, retriever, and prompt in the codebase | Live walkthrough |

### 5.3 Process / deliverables

| # | Criterion |
|---|---|
| P1 | Chunking strategy documented with **justification**, chunk size, overlap, and retained metadata — written *before* implementation |
| P2 | `README.md` with setup steps, scope, and known limits |
| P3 | `docs/sources.md` lists the 5 URLs |
| P4 | Sample Q&A file with 5–10 queries, answers, and links |
| P5 | Disclaimer string in the UI matches the README/snippet exactly |
| P6 | `.env` is git-ignored; no key in any committed file |
| P7 | ≤3-min demo video recorded as backup to hosting |

### 5.4 Explicit failure (counts against the milestone)

- Any fabricated number presented as fact
- Any buy/sell/return recommendation leaking into an answer
- A PII value stored anywhere, including logs
- A missing or non-allowlisted citation
- Key committed to Git

---

## 6. Constraints

### 6.1 Hard constraints (from the brief — non-negotiable)

| ID | Constraint | Impact on design |
|---|---|---|
| C1 | **Free-tier tools only** | ChromaDB, MiniLM, and all Python libs are free/open-source. Groq free tier provides the LLM. No paid embedding API, no hosted vector DB, no LangSmith, no paid tracing. |
| C2 | **Runs locally** | App must start with one command on a laptop, offline except for the Groq call and the initial page fetch. No dependency on Render, a container registry, or a cloud DB to develop. |
| C3 | **Deployable to Render** | Must fit a free web service: `buildCommand`, `startCommand`, `PORT` binding, health check, and no long blocking build. Render free tier = ephemeral filesystem and sleep after inactivity → see C4/C5. |
| C4 | **Public sources only** | Fixed 5-URL allowlist. No third-party blogs, no app back-end screenshots. Citations restricted to the allowlist (+ official educational pages for refusals). |
| C5 | **No PII** | No inputs or storage of PAN, Aadhaar, account numbers, OTPs, emails, phone numbers. No auth, no user accounts, no chat-log persistence. |
| C6 | **No performance claims** | Never compute or compare returns. If asked, link to the official factsheet. |
| C7 | **Embedding model is fixed** | `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, local, no API key — the **same** model for chunks and queries. ChromaDB collection must therefore be `384`-dimensional. |
| C8 | **Vector DB is fixed** | ChromaDB, persisted to disk, single ingestion run. |
| C9 | **LLM is fixed** | Groq, key in `.env`, never committed. |
| C10 | **Chunking is agent-decided, not arbitrary** | Must inspect the data, propose size/overlap/metadata with reasoning, and save chunks to a readable `.txt` before/alongside implementation. |
| C11 | **Answer brevity** | ≤ 3 sentences, one link, "Last updated from sources:". |
| C12 | **State management** | Stateless chat. No cross-session memory, no user profiles. |

### 6.2 Engineering constraints

- **Python** for app, ingestion, and evaluation (FastAPI or Streamlit for the UI; both are free-tier and Render-friendly).
- **One-command local start**: `pip install -r requirements.txt` → `python ingest.py` (once) → `uvicorn main:app` or `streamlit run app.py`.
- **Graceful degradation**: if the Groq key is missing, the app must fail with a clear setup message rather than a stack trace.
- **Retrieval guardrail**: a similarity-score threshold below which the bot says "not in my sources" instead of answering — this is what makes F10 pass.
- **Prompt-level refusals** backed by an **output check** (e.g. regex for performance/advice language) so violations are caught in testing, not just hoped for.

### 6.3 Operational / deployment constraints

- **Render free tier has an ephemeral filesystem.** The Chroma store must be built during the build or on first boot and cached (e.g. in a build step or a Render Disk if available), and the app must start serving even if the store is missing by re-ingesting automatically. This is the single biggest deployment risk — decide and document the approach before deploying.
- **Cold starts** (free tier sleeps) mean the first request may take 30–50 s. The UI must show a loading state; the demo video should be the safety net.
- **Corpus freshness is manual.** The 5 pages are a point-in-time snapshot with a "Last updated" date. The README must state this and give a re-ingest command.
- **No secrets in the repo.** `.env` + `.gitignore`; deploy-time env var set in the Render dashboard.

### 6.4 Known limits to disclose in the README

1. Answers are only as good as the 5 pages; anything not on them is out of scope by design.
2. Only Direct–Growth variants are covered — Direct–IDCW, Regular, and other AMCs are not.
3. Facts are a snapshot date and may be stale (expense ratios and exit loads change).
4. No NAV/live pricing — the pages are static.
5. Natural-language matching can miss paraphrased questions; there is no synonym/keyword layer.
6. English only.
7. Small corpus (~5 pages) means retrieval is easier than at scale — the demo does not stress-test ranking quality.

---

## 7. Decisions on Former Open Questions

Resolved during the data-inspection pass. Full reasoning in `docs/CHUNKING.md` §6.

| # | Question | Decision |
|---|---|---|
| Q1 | UI framework | **FastAPI + one static HTML file.** Precise control over disclaimer/citation rendering (F2/F4), fast cold start on Render, native `/health`. Streamlit is the fallback. |
| Q2 | Chroma persistence on Render | **Ingest at build time, auto re-ingest on boot if the store is missing**, raw HTML cached in `data/raw/`, `content_hash` idempotence guard. |
| Q3 | HTML cleaning approach | **BeautifulSoup + heading-text section allowlist.** CSS classes are content-hashed and brittle. |
| Q4 | Retrieval `k` and threshold | **`k=5`, cosine distance, scheme metadata filter**, threshold tuned on the 10-query eval set then frozen and documented. |
| Q5 | Groq model and token limit | **`temperature=0`, `max_tokens=1024`**, model id via env var (free-tier model availability changes). *Deviation from the brief's `≈250`: the selected model is a reasoning model and its reasoning tokens count against this budget, so 250 returned truncated answers (3/8 runs measured). 1024 measured 0/8. The ≤3-sentence rule is enforced by output validation, not by this number.* |

### 7.1 Risks found during inspection (change the plan)

Data inspection turned up three problems that constrain the build — full detail in `docs/CHUNKING.md` §1:

1. **The source pages contain returns, CAGR, and category rankings.** Chunking whole pages would put performance claims inside the vector store, breaching C6/F7. Exclusion must happen at extraction, not in the prompt.
2. **Every page embeds data for many other schemes** (82 scheme codes on the Balanced Advantage page). Chunks must be filtered to the page's own `scheme_code`, or answers will cite the right URL for the wrong scheme.
3. **`nfo_risk` is boilerplate that contradicts the page prose** ("Moderately High" vs "Very High" on all five). It is denylisted.

---

## 8. Acceptance Checklist

- [ ] Pipeline runs end to end: Load → Chunk → Embed → Store, then Question → Embed → Retrieve → LLM → Answer
- [ ] `chunks.txt` committed and inspectable
- [ ] Chroma store persists across restarts
- [ ] 3 welcome questions answer correctly
- [ ] Every answer: ≤3 sentences, one link, "Last updated from sources"
- [ ] Refusal path works on all 6 opinionated questions
- [ ] PII inputs refused, not stored
- [ ] No performance claims anywhere
- [ ] Disclaimer visible in UI and matching the snippet
- [ ] `.env` ignored, key not committed
- [ ] `README.md` + `docs/sources.md` + sample Q&A file complete
- [ ] Deploys to Render free tier and stays up, or ≤3-min demo video recorded
