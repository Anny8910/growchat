# Chunking Strategy Proposal — Mutual Funds RAG Chatbot

**Status:** Proposal, written **before** implementation (required by constraint C10)
**Date:** 2026-09-29
**Informs:** `docs/PRD.md` §3.2, §6.2

This document reports what the actual corpus looks like, proposes a chunking strategy, justifies it against that data, and specifies chunk size, overlap, and retained metadata.

---

## 1. Data inspection

All 5 URLs were fetched and inspected before proposing anything.

| Scheme | HTTP | Raw HTML | Text after readability extraction | Own `scheme_code` |
|---|---|---|---|---|
| HDFC Large Cap – Direct Growth | 200 | 454 KB | 486 words | `119018` |
| HDFC Equity (Flexi Cap) – Direct Growth | 200 | 494 KB | 492 words | `118955` |
| HDFC ELSS Tax Saver – Direct Growth | 200 | 452 KB | 433 words | `119060` |
| HDFC Small Cap – Direct Growth | 200 | 497 KB | 496 words | `130503` |
| HDFC Balanced Advantage – Direct Growth | 200 | 815 KB | 554 words | `118968` |

**Finding 0 — the pages are client-rendered but the data is in the HTML.**
A naive "fetch and hope" would return an empty shell. However every fact we need is present in the raw response, in two forms: rendered prose/tables, and a JSON payload embedded in the page. So the loader does not need a headless browser. *(Free-tier constraint C1 respected — no Browserless, no ScrapingBee, no paid renderer.)*

### Finding 1 — "readability extract the whole page" is unsafe. It leaks performance claims.

Standard readability extraction of the ELSS page yields:

```
| 1 year | ₹60,000 | ₹57,266 | -4.56% |
| 3 years | ₹1,80,000 | ₹1,85,328 | +2.96% |
| 5 years | ₹3,00,000 | ₹3,88,389 | +29.46% |
| 10 years | ₹6,00,000 | ₹12,09,718 | +101.62% |
```

The page also embeds structured "pros/cons" records with these `analysis_subject` values:

| Subject | Pages | Contains performance claim? |
|---|---|---|
| `expense_analysis` | 5 | No — expense ratio |
| `base_expense_analysis` | 5 | No — base expense |
| `exit_load` | 1 | No — exit load |
| `lock_in` | 1 | No — lock-in |
| `credit_rating_analysis` | 1 | No — holdings rating |
| `c_return_analysis` | 5 | **Yes** — "consistently lower annualised returns than category average" |
| `p_return_analysis` | 3 | **Yes** — "5Y and 10Y annualised returns higher than category average" |
| `alpha_analysis` | 2 | **Yes** — "fund has generated returns higher than benchmark" |

**Consequence:** chunking the whole page would put returns, CAGR, rankings, and category comparisons into the vector store, where retrieval can surface them and the LLM can quote them. That directly violates constraint **C6 (no performance claims)** and success criterion **F7**. *Exclusion is a corpus-construction requirement, not a prompt requirement.* Prompts are not a reliable control here; the data must not contain the claim in the first place.

### Finding 2 — every page embeds data for *other* schemes. Cross-scheme contamination is real.

Distinct `scheme_code` values found per page:

| Page | Distinct scheme codes | Own scheme's share |
|---|---|---|
| Large Cap | 46 | `119018` × 62 |
| Flexi Cap | 46 | `118955` × 100 |
| ELSS | 42 | `119060` × 79 |
| Small Cap | 50 | `130503` × 99 |
| Balanced Advantage | **82** | `118968` × 348 |

The Balanced Advantage page carries 82 scheme codes (peer/comparison widgets). A whole-page chunk would answer "what is the expense ratio of HDFC Large Cap?" with Balanced Advantage content, and the citation would still look correct — the most damaging possible failure, because it defeats F2's credibility.

**Mitigation:** a chunk is only admitted if its `scheme_code` equals the page's own scheme code.

### Finding 3 — boilerplate contradicts the real data. `nfo_risk` must be dropped.

| Page | `nfo_risk` field (JSON) | Prose on page |
|---|---|---|
| Large Cap | `Moderately High Riskometer` | "rated Very High risk" |
| Flexi Cap | `Moderately High` | "rated Very High risk" |
| ELSS | `Moderately High` | "rated Very High risk" |
| Small Cap | `Moderately High Riskometer` | "rated Very High risk" |
| Balanced Advantage | `Moderately High Riskometer` | "rated Very High risk" |

`nfo_risk` is a **template leftover, identical across all five schemes**, and it disagrees with the page's own prose. Embedding it would give the bot a confidently wrong riskometer for every scheme — the exact "fabricated number" failure listed in §5.4 of the PRD.

**Mitigation:** denylist the `nfo_risk` field; take riskometer from the scheme summary prose only.

### Finding 4 — facts are stored as *disconnected* label/value pairs.

The rendered page gives `Expense ratio` and `1.21%` as separate lines, and `Min. for SIP` / `₹500` likewise. Chunked as-is, the number is separated from its label and becomes unattributable. Exit loads are worse — they are date-keyed rows in a popup:

```
08 May 2015 | Exit load of 1% if redeemed within 1 year
16 Feb 2015 | Exit load of 1% if redeemed within 18 months
```

and one scheme has no exit load row at all. **Label–value and row–header pairs must be rejoined before chunking.**

### Finding 5 — the embedding model caps chunk length.

`sentence-transformers/all-MiniLM-L6-v2` has `max_seq_length = 256` word-pieces. Any chunk beyond that is **silently truncated** — losing the tail of a table, which is usually where the answer is. Chunk sizing must be expressed against this limit, not chosen by feel.

---

## 2. Proposed strategy

**Name:** *Structure-first, allowlist extraction with pair re-joining, then size-capped packing.*

Four stages. Stages 1–3 are what make this corpus safe; stage 4 is what makes it retrieve well.

### Stage 1 — Load
- `requests` with a browser `User-Agent` (plain `urllib` gets blocked).
- Save raw HTML to `data/raw/<slug>.html` and record `fetched_at`. Caching makes re-ingestion offline-capable, gives us the "Last updated from sources" date for free, and makes builds on Render reproducible.
- Fail loudly and list the URL if a fetch returns non-200. Silently ingesting a shell page would produce a bot that confidently answers nothing.

### Stage 2 — Section-aware extraction (allowlist, not denylist)
Extract **named sections only**, located by their visible heading text, not by CSS class (the classes are content-hashed, e.g. `exitLoadStampDutyTax_popupBody__4ywku`, and will break without notice):

| Section | Answers | Source form |
|---|---|---|
| Scheme overview / about | launch date, AUM, risk rating, min SIP, min lump sum | prose |
| Fund benchmark | benchmark index | key–value grid |
| Exit load | tiered slabs by date | date-keyed table |
| Fees / expense | expense ratio, base expense | key–value + JSON record |
| ELSS lock-in | lock-in period | JSON record `analysis_subject: lock_in` |
| Tax / stamp duty | tax treatment on redemption | prose |
| Download statement guide | how-to steps | guide text |

**Admitted only if** the enclosing block's `scheme_code` matches the page's own scheme code (fixes Finding 2).

**Never admitted:** the returns/rankings tables, `c_return_analysis` / `p_return_analysis` / `alpha_analysis` records, `nfo_risk`, nav/footer/FAQ-boilerplate, peer-scheme widgets (fixes Findings 1–3).

Allowlist over denylist is deliberate: a denylist of "performance" keywords fails on the *next* page layout, and the failure mode is silent and compliance-relevant. An allowlist fails safe — new content is simply invisible until someone adds it.

### Stage 3 — Normalise into fact records, then re-join
Each extracted section becomes a set of records, each with a stable `field` key:

```
field: exit_load     value: "1% if redeemed within 1 year"     as_of: 2026-09-28
field: expense_ratio value: "1.21%"                            as_of: 2026-09-28
field: lock_in       value: "3 years"                          as_of: 2026-09-28
```

Then pack them into a chunk **with their field labels inline**, so the embedded text is self-describing:

> **HDFC ELSS Tax Saver Fund – Direct Growth** (HDFC Mutual Fund)
> **Lock-in period:** 3 years. **Expense ratio:** 1.21%. **Minimum SIP:** ₹500.
> **Benchmark:** NIFTY 500 Total Return Index.
> Source: https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth

This is what makes the retrieved text unambiguous to both the embedder and the LLM (fixes Finding 4). Chunks are prefixed with the scheme name because "expense ratio" alone is meaningless across five schemes sharing an embedding space.

### Stage 4 — Size-capped packing
- **Target: 150 words ≈ 200 word-pieces** — comfortably under the 256 limit, leaving headroom for the prefix and for the money symbol `₹` and accented text tokenising more finely than plain ASCII.
- **Overlap: 25 words (~17%)** — applies **only between prose chunks**. A fact record is atomic; splitting `expense_ratio: 1.21%` from its label is the one failure mode that produces a confidently wrong number. Prose (tax treatment, how-to steps) gets overlap because sentences straddle boundaries there; fact groups get overlap only if a group is split.
- **Never pack across schemes.** Packing is per-page, so every chunk's metadata has exactly one scheme.
- **No packing across topics.** One section per chunk keeps citations precise: an exit-load answer cites a chunk that contains only exit load.

### Retrieval-side counterpart
`k = 5`, cosine distance, plus a **metadata filter on `scheme`** when the question names a scheme, so a query about HDFC ELSS can never retrieve a Small Cap chunk. Below-threshold results trigger the "not in my sources" path (PRD F10). `k` and the threshold get tuned against the 10-query eval set, then frozen and written into the README.

---

## 3. Chunk size, overlap, metadata — the required summary

| Parameter | Value | Why |
|---|---|---|
| Chunk size | 150 words (~200 tokens) | Under the 256-token MiniLM limit, so no silent truncation (Finding 5) |
| Overlap | 25 words, prose only | Protects sentence-spanning claims; fact records never split |
| Strategy | Structure-first, allowlist | Only way to exclude returns reliably (Findings 1–3) |
| Split unit | Section, not fixed window | Keeps a citation semantically single-topic |
| Scheme isolation | One scheme per chunk + metadata filter | 82 scheme codes on one page (Finding 2) |
| Booterplate | `nfo_risk` denylisted | Contradicts page prose (Finding 3) |

### Metadata schema (one record per chunk)

| Field | Example | Used for |
|---|---|---|
| `source_url` | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` | The single citation (F2) |
| `scheme_name` | `HDFC ELSS Tax Saver Fund – Direct Growth` | Filtering, disambiguation |
| `scheme_code` | `119060` | Contamination guard (Finding 2) |
| `category` | `Equity ELSS` | Routing |
| `section` | `fees` / `exit_load` / `lock_in` / `benchmark` / `risk` | Precision filtering |
| `fields` | `["expense_ratio","min_sip"]` | Test assertions |
| `chunk_index` | `3` | Ordering/debug |
| `retrieved_at` | `2026-09-28` | "Last updated from sources" (F4) |
| `content_hash` | `sha256:…` | Re-ingest idempotence |

`chunks.txt` is written with all of the above visible per chunk, as required by C10/F9.

---

## 4. Expected result

Roughly **35–55 chunks** across 5 pages (~7–11 per page: overview, benchmark, exit load, fees, risk, min-investment, tax, and ELSS-only lock-in). Small by design — a 5-page corpus is a demo, and PRD §6.4 discloses that it does not stress-test ranking.

---

## 5. Risks in this proposal

| Risk | Mitigation |
|---|---|
| Heading text changes on groww.in | Section allowlist is a single config dict; missing sections fail the build rather than silently shrinking the corpus |
| Exit-load popups are lazily mounted | Verified present in raw HTML for all 5; re-verify at ingest and alert if a section goes missing |
| `trafilatura` on Python 3.9 is broken (`lxml.html.clean` split out) | Use BeautifulSoup + explicit section selection; `trafilatura` is a fallback for prose only, with `lxml_html_clean` pinned |
| MiniLM is a 2023-era model on financial phrasing | Acceptable for 5 pages; compensated by field labels inline in chunk text |
| Threshold tuning is subjective | Freeze on the 10-query set and record the numbers in the README |

---

## 6. Decisions taken on the PRD's open questions

| # | Decision | Reasoning |
|---|---|---|
| **Q1** | **FastAPI + one static HTML file**, not Streamlit | Precise control over the disclaimer and citation rendering (F2/F4 are hard requirements); a tiny static payload starts fast on Render's free tier; native `/health` endpoint. Streamlit is the fallback if the UI work threatens the milestone — the pipeline is decoupled behind a `rag.answer()` function either way |
| **Q2** | **Ingest at build time on Render**, with auto re-ingest on boot if the store is missing; raw HTML cached in `data/raw/` | Render's free tier has no persistent disk, so the store must live in the built image. Idempotence guard via `content_hash` prevents repeated work |
| **Q3** | **BeautifulSoup + heading-text section allowlist** (see Stage 2) | CSS classes are content-hashed and brittle; heading text is what a human sees |
| **Q4** | **`k=5`, cosine distance, scheme metadata filter, tuned threshold** — to be frozen after the eval set | See §2 |
| **Q5** | **Groq, `temperature=0`, `max_tokens=1024`; model id via env var** | The budget covers the model's reasoning tokens as well as the answer, so it is set well above the 3-sentence answer length; the ≤3-sentence rule is enforced at validation, not by this number. Free-tier model availability changes — the model id is env-configurable rather than hardcoded, and is verified at implementation time |
