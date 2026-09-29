# Architecture — Mutual Funds Facts-Only RAG Chatbot

**Version:** 1.0
**Date:** 2026-09-29
**Derived from:** `docs/PRD.md`, `docs/CHUNKING.md`
**Status:** Ready for implementation

---

## 1. Architecture at a glance

A single Python process, run locally and deployed as one Render web service. Two pipelines:

- **Ingestion** (`ingest.py`) — runs once, offline-capable, writes three artefacts: raw HTML cache, `chunks.txt`, and a persisted ChromaDB store.
- **Query** (`app.py`) — runs per request, stateless, never re-embeds the corpus.

The only network dependency at query time is the Groq API. The embedding model runs in-process.

```
┌────────────────────────── INGESTION (once) ──────────────────────────┐
│                                                                      │
│  5 URLs ──► loader ──► extractor ──► chunker ──► embedder ──► store   │
│            (cache)   (allowlist)   (chunks.txt) (MiniLM)   (ChromaDB) │
│                                                                      │
│  emits:  data/raw/*.html   data/chunks.txt   data/chroma/            │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼  (read-only, shared)
┌─────────────────────────── QUERY (per request) ──────────────────────┐
│                                                                      │
│  question ──► guardrails ──► retriever ──► generator ──► answer      │
│               (PII/advice)  (embed+top-k) (Groq, temp=0)  + citation │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Components

### 2.1 Ingestion components

| # | Component | Module | Responsibility | Key rule it enforces |
|---|---|---|---|---|
| I1 | **Loader** | `rag/loader.py` | Fetch the 5 allowlisted URLs with a browser UA; cache raw HTML to `data/raw/`; record `fetched_at` | Fails loudly on non-200 — a silently-ingested JS shell produces a bot that confidently answers nothing |
| I2 | **Extractor** | `rag/extract.py` | Section allowlist keyed on visible heading text → structured fact records (`field`, `value`, `as_of`) | Drops returns/NAV-performance tables, `c_return_analysis` / `p_return_analysis` / `alpha_analysis`, `nfo_risk`, nav/footer, peer-scheme widgets (PRD §7.1) |
| I2a | **Scheme guard** | `rag/extract.py` | A record is admitted only if its `scheme_code` equals the page's own code | 82 scheme codes appear on the Balanced Advantage page; without this, the right URL gets cited for the wrong scheme |
| I3 | **Chunker** | `rag/chunker.py` | Re-join label/value pairs into self-describing text; pack per section, one scheme per chunk; emit metadata; write `chunks.txt` | 150-word cap (~200 tokens, under MiniLM's 256 limit) so nothing is silently truncated |
| I4 | **Embedder** | `rag/embedder.py` | `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, CPU, singleton | Same model instance/mode for chunks **and** queries — mismatch makes cosine distance meaningless (C7) |
| I5 | **Store** | `rag/store.py` | ChromaDB `PersistentClient`, collection `hdfc_faqs`, cosine space, 384-dim | Written once; queried forever (C8, F8) |

### 2.2 Query components

| # | Component | Module | Responsibility | Key rule it enforces |
|---|---|---|---|---|
| Q1 | **API** | `app.py` | FastAPI. `POST /api/ask`, `GET /api/examples`, `GET /health`. Serves `static/index.html` | Stateless; no chat history, no user accounts, no PII persistence (C5, C12) |
| Q2 | **Guardrails** | `rag/guardrails.py` | (a) PII detection; (b) advice/opinion classification; (c) post-generation output validation | Refuses before the LLM is ever called; validates the answer *after* it is generated, so a violation is caught in testing rather than hoped for |
| Q3 | **Retriever** | `rag/retriever.py` | Embed question with MiniLM → `k=5` cosine search → optional `scheme` metadata filter → distance threshold | Below threshold returns "not in my sources" + link instead of a guess (F10) |
| Q4 | **Generator** | `rag/generator.py` | Build the grounded prompt + answer contract; call Groq (`temperature=0`, `max_tokens=1024`); attach citation and `Last updated from sources` | ≤3 sentences, exactly one citation, no advice, no performance claims (C6, C11) |
| Q5 | **UI** | `static/index.html` | One file, vanilla JS. Welcome line, 3 clickable examples, chat thread, citation per answer, persistent disclaimer, loading state | Disclaimer string matches the README snippet exactly (P5) |

### 2.3 Configuration

| Component | Module | Responsibility |
|---|---|---|
| **Settings** | `config.py` | Loads `SOURCES`, `EMBED_MODEL`, `CHROMA_DIR`, `CHUNK_WORDS`, `CHUNK_OVERLAP`, `TOP_K`, `DISTANCE_THRESHOLD`, `GROQ_MODEL`, `GROQ_API_KEY` from env with defaults. Fails with a clear setup message when the key is missing — not a stack trace. |

---

## 3. Data flow

### 3.1 Ingestion — `python ingest.py`

Runs once. Idempotent: re-running with unchanged `content_hash` is a no-op.

| Stage | Module | Input | Output | Guard |
|---|---|---|---|---|
| **Load** | `loader.py` | 5 URLs | `data/raw/<slug>.html` + `fetched_at` | HTTP 200 required; browser UA; cache hit skips network |
| **Extract** | `extract.py` | Raw HTML | List of `FactRecord(field, value, as_of, section)` | Section allowlist; `scheme_code` must match page; performance regions and `nfo_risk` dropped |
| **Chunk** | `chunker.py` | Fact records | List of `Chunk(text, metadata)` | ≤150 words; 25-word overlap on prose only; never split a field from its label; one scheme per chunk; one section per chunk |
| **Embed** | `embedder.py` | Chunk texts | 5 × 384-dim float vectors | MiniLM, CPU, `normalize_embeddings=True` |
| **Store** | `store.py` | Vectors + metadata | `data/chroma/` (persistent) | Cosine space; `content_hash` for idempotence |
| **Dump** | `chunker.py` | Chunks | `data/chunks.txt` | Every chunk with full metadata visible, human-readable (C10, F9) |

### 3.2 Query — `POST /api/ask`

| Stage | Module | What happens | Early exit |
|---|---|---|---|
| 1. **Guard** | `guardrails.py` | PII pattern match; then advice/intent classification | PII or advice → refusal response, **no LLM call** |
| 2. **Embed** | `retriever.py` + `embedder.py` | Question → 384-dim vector (same MiniLM) | — |
| 3. **Retrieve** | `retriever.py` | Chroma `query(n_results=5, where={scheme})` | Best distance > threshold → "not in my sources" + link |
| 4. **Generate** | `generator.py` | Grounded prompt + context chunks + answer contract → Groq, `temperature=0`, `max_tokens=1024` | Groq failure → explicit error, never a fabricated answer |
| 5. **Validate** | `guardrails.py` | Sentence count, single citation from the 5-URL allowlist, advice/performance regex | Violation → replaced with a safe fallback response |
| 6. **Compose** | `generator.py` | `answer`, `source_url`, `retrieved_at`, `disclaimer` | — |

### 3.3 Query flow diagram

```
                    ┌──────────────────────────────────────┐
   Browser          │  static/index.html                   │
   (no state)       │  welcome + 3 examples + disclaimer    │
                    └───────────────┬──────────────────────┘
                                    │  POST /api/ask {question}
                                    ▼
                    ┌──────────────────────────────────────┐
                    │  app.py  (FastAPI, stateless)         │
                    └───────────────┬──────────────────────┘
                                    ▼
                    ┌──────────────────────────────────────┐
                    │  guardrails.check_pii()               │
                    └───┬──────────────────────────┬───────┘
                 PII yes│                          │no
                        ▼                          ▼
              ┌───────────────────┐   ┌──────────────────────────┐
              │ Refuse, PII not   │   │ guardrails.is_advice()   │
              │ echoed or logged  │   └───┬──────────────┬───────┘
              └─────────┬─────────┘  advice│              │factual
                          │                ▼              ▼
                          │   ┌──────────────────┐  (continue)
                          │   │ Refuse +         │
                          │   │ educational link │
                          │   └────────┬─────────┘
                          │            │
                          │            ▼
                          │  ┌───────────────────────────────┐
                          │  │ retriever.retrieve()          │
                          │  │  1. embed q (MiniLM, 384-dim) │
                          │  │  2. chroma.query(k=5, cosine) │
                          │  │     where={scheme: ...}        │
                          │  │  3. distance > threshold?      │
                          │  └───┬───────────────────┬───────┘
                          │   yes │                   │ no
                          │       ▼                   ▼
                          │  ┌───────────────┐  ┌──────────────────────┐
                          │  │ "Not in my    │  │ generator.generate() │
                          │  │ sources" +   │  │  grounded prompt     │
                          │  │ link         │  │  + answer contract   │
                          │  └───────┬───────┘  │  Groq temp=0         │
                          │          │          │  max_tokens=1024     │
                          │          │          └──────────┬───────────┘
                          │          │                     ▼
                          │          │          ┌──────────────────────┐
                          │          │          │ guardrails.          │
                          │          │          │ validate_output()    │
                          │          │          │ ≤3 sent · 1 link    │
                          │          │          │ allowlist · no advice│
                          │          │          └──────────┬───────────┘
                          │          │                     ▼
                          └──────────┴────────► ┌──────────────────────┐
                                                 │ { answer,            │
                                                 │   source_url,        │
                                                 │   retrieved_at,      │
                                                 │   disclaimer }       │
                                                 └──────────┬───────────┘
                                                            ▼
                                                 rendered with clickable
                                                 citation + "Last updated"
```

---

## 4. Tech stack

| Layer | Choice | Why | Constraint served | Cost |
|---|---|---|---|---|
| Language | **Python 3.11+** | Brief specifies it; best library coverage for embeddings | C1, C2 | Free |
| HTTP client | `requests` | Simple, handles UA/retries; `urllib` gets blocked by groww.in | C1 | Free |
| HTML parsing | `beautifulsoup4` + `lxml` | Heading-text section allowlist; CSS classes are content-hashed and break silently | C1, Q3 | Free |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, ~90 MB, CPU, no API key. Fixed by brief | **C7** | Free |
| Vector store | **ChromaDB** (`PersistentClient`, cosine, 384-dim) | Fixed by brief; in-process, zero server to run | **C8** | Free |
| LLM | **Groq** (`temperature=0`, `max_tokens=1024`, model id via env) | Fixed by brief; fast, free tier. `temperature=0` for factual determinism. `max_tokens` covers the model's reasoning tokens as well as the answer; the ≤3-sentence rule is enforced at validation | **C9, C11** | Free tier |
| Web framework | **FastAPI** + `uvicorn` | Precise control of citation/disclaimer rendering, tiny cold start, native `/health` | C3, Q1 | Free |
| Frontend | Vanilla HTML/CSS/JS, single file | No build step, no framework, no CDN dependency | C1, C2 | Free |
| Config/secrets | `python-dotenv` + `.gitignore` | Key in `.env`, never committed | C9, P6 | Free |
| Hosting | **Render** free web service | Required target; `buildCommand` runs ingestion, `startCommand` serves uvicorn | **C3** | Free tier |

**Not used, deliberately:** no LangChain/LlamaIndex (the pipeline is 6 small functions and a framework would obscure exactly what the evaluator needs to see, PRD Q5), no hosted vector DB, no paid embedding API, no tracing/telemetry service, no headless browser (all data is present in the raw HTML), no auth.

### Deployment split — build time vs runtime

| | Build time (`buildCommand`) | Runtime (`startCommand`) |
|---|---|---|
| Runs | `pip install` + `python ingest.py` | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Network | Fetches 5 pages, embeds ~40 chunks | Groq only |
| Writes | `data/raw/`, `data/chunks.txt`, `data/chroma/` | Nothing |
| On Render free tier | Artefacts baked into the image (no persistent disk) | If `data/chroma/` is missing, auto re-ingest on boot, guarded by an idempotence check |

This is the mitigation for the ephemeral-filesystem risk called out in PRD §6.3. **Pinning versions in `requirements.txt` matters here** — a dependency change between build and run would otherwise produce a dimension mismatch against the persisted store.

---

## 5. Folder structure

```
m4-growchat/
├── README.md                    # setup, scope, known limits (P2)
├── requirements.txt             # pinned
├── .env                         # GROQ_API_KEY — git-ignored, never committed (P6)
├── .gitignore                   # .env, data/chroma/, __pycache__/
├── render.yaml                  # build/start commands, health check
├── ingest.py                    # CLI entry: Load → Extract → Chunk → Embed → Store
├── app.py                       # FastAPI: /api/ask, /api/examples, /health
├── config.py                    # env-driven settings + defaults
│
├── rag/
│   ├── loader.py                # I1  fetch + cache raw HTML
│   ├── extract.py               # I2  section allowlist → fact records (+ scheme guard)
│   ├── chunker.py               # I3  pair re-joining, packing, chunks.txt
│   ├── embedder.py              # I4  MiniLM singleton (chunks + queries)
│   ├── store.py                 # I5  ChromaDB persistent client
│   ├── retriever.py             # Q3  embed → top-k → threshold → scheme filter
│   ├── generator.py             # Q4  grounded prompt → Groq → answer + citation
│   ├── guardrails.py            # Q2  PII, advice classification, output validation
│   └── prompts.py               # system prompt + answer contract, versioned in one place
│
├── static/
│   └── index.html               # Q5  entire UI, one file
│
├── data/                        # git-ignored except chunks.txt
│   ├── raw/                     # cached HTML (offline re-ingest)
│   ├── chunks.txt               # COMMITTED — human-readable, required by C10/F9
│   └── chroma/                  # persisted vector store
│
├── tests/
│   ├── test_extraction.py       # no performance fields; no cross-scheme records
│   ├── test_chunking.py         # size cap, one scheme per chunk, no split fields
│   ├── test_guardrails.py       # 6 refusals + 3 PII cases
│   └── test_api.py              # /health, /api/ask contract shape
│
└── docs/
    ├── problemstatement.txt
    ├── PRD.md
    ├── CHUNKING.md              # pre-implementation strategy (C10)
    ├── architecture.md          # this file
    ├── sources.md               # the 5 URLs (P3)
    └── sample_qa.md             # 5–10 queries + answers + links (P4)
```

**Two things are committed that normally are not:** `docs/*` and `data/chunks.txt`. The evaluator must be able to read the chunk file without running anything, and `chunks.txt` is a graded deliverable (F9).

---

## 6. Data contracts

### 6.1 FactRecord (extractor → chunker)

```python
@dataclass
class FactRecord:
    scheme_code: str    # "119060"
    scheme_name: str    # "HDFC ELSS Tax Saver Fund – Direct Growth"
    section: str        # "fees" | "exit_load" | "lock_in" | "benchmark" | ...
    field: str          # "expense_ratio" | "min_sip" | "lock_in" | ...
    value: str          # "1.21%"
    as_of: str          # "2026-09-28"
    source_url: str     # one of the 5 allowlisted URLs
```

### 6.2 Chunk (chunker → embedder → store)

`text` is self-describing — the scheme name and field labels are inline, so the embedded text is unambiguous across five schemes sharing one vector space:

> **HDFC ELSS Tax Saver Fund – Direct Growth** (HDFC Mutual Fund)
> **Lock-in period:** 3 years. **Expense ratio:** 1.21%. **Minimum SIP:** ₹500.
> **Benchmark:** NIFTY 500 Total Return Index.

`metadata` per chunk: `source_url`, `scheme_name`, `scheme_code`, `category`, `section`, `fields[]`, `chunk_index`, `retrieved_at`, `content_hash` (full schema in `docs/CHUNKING.md` §3).

### 6.3 API response

```json
{
  "answer": "The minimum SIP for HDFC Balanced Advantage Fund – Direct Growth is ₹500.",
  "source_url": "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
  "retrieved_at": "2026-09-28",
  "disclaimer": "Facts-only. No investment advice.",
  "refused": false
}
```

`refused: true` covers PII, advice, and "not in my sources" — the UI renders one consistent panel for all three, with the educational link.

---

## 7. How the architecture satisfies the constraints

| Constraint | How |
|---|---|
| **C1** free-tier only | Every component is an open-source library or a free tier; nothing added to this table was paid for |
| **C2** runs locally | `pip install -r requirements.txt` → `python ingest.py` → `uvicorn app:app`. No container, no external DB, no cloud dependency |
| **C3** Render-deployable | Static build/start commands, `PORT` binding, `/health`; ingestion inlined into the build to sidestep the ephemeral disk |
| **C4** public sources only | The 5 URLs live in `config.py` as the single allowlist; `guardrails.validate_output` rejects any citation outside it |
| **C5** no PII | PII check runs **before** the LLM; the matched value is never logged or echoed; no persistence of any kind in the query path |
| **C6** no performance claims | Enforced in the **extractor** (performance regions never become chunks), reinforced in the prompt, verified in `validate_output` |
| **C7** fixed embedding model | One `embedder.py` singleton used by both ingestion and retrieval; Chroma collection fixed at 384-dim |
| **C8** Chroma persisted to disk | `PersistentClient`; single ingestion run; `content_hash` idempotence |
| **C9** Groq via `.env` | `config.py` reads `GROQ_API_KEY`; `.gitignore` blocks `.env`; missing key yields a setup message |
| **C10** chunking decided with justification | `docs/CHUNKING.md`, written before code, grounded in inspection of the real pages |
| **C11** ≤3 sentences, one link | Sentence-count and allowlist checks at validation (the rule is never left to the token budget) |
| **C12** stateless | No sessions, no history, no user records; every request is independent |

---

## 8. Failure modes and handling

| Failure | Handling | User sees |
|---|---|---|
| Groq key missing at startup | Config raises a setup error | Clear "add `GROQ_API_KEY` to `.env`" message, not a stack trace |
| Groq API error / rate limit (free tier) | Caught, retried once, then explicit failure | "I'm temporarily unable to answer — try again." Never a fabricated answer |
| Question outside the corpus | Distance > threshold at stage 3 | "I don't have that in my sources" + relevant link |
| Opinionated or PII question | Guardrail at stage 1, before any LLM call | Polite facts-only refusal + educational link |
| Chroma store missing on boot | Auto re-ingest, then serve | Longer first-load, then normal |
| Source page changed / fetch fails | Loader raises with the URL | Ingestion aborts loudly rather than ingesting a shell page |
| LLM produces >3 sentences or a bad citation | `validate_output` rejects | Safe fallback response, logged for the eval set |

**Principle:** every failure path degrades to "I don't know" or a refusal. No path degrades to a guess. That is what keeps a hallucinated number off the screen during the demo.

---

## 9. Implementation order

1. `config.py` + `rag/loader.py` + `rag/extract.py` → verify `chunks.txt` by eye
2. `rag/chunker.py` → confirm 150-word cap, one scheme per chunk, no split fields
3. `rag/embedder.py` + `rag/store.py` → `python ingest.py`, confirm the store persists
4. `rag/retriever.py` → check the 10 eval queries retrieve the right chunk
5. `rag/prompts.py` + `rag/generator.py` + `rag/guardrails.py` → answer contract and refusals
6. `app.py` + `static/index.html` → the tiny UI
7. `render.yaml` + deploy; record the ≤3-min demo video
8. `README.md`, `docs/sources.md`, `docs/sample_qa.md`

Steps 1–3 are verifiable without an API key, so the pipeline can be proven working before the LLM is wired in.
