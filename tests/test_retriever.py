"""Retrieval tests: scheme detection, top-k correctness, out-of-corpus, threshold.

The retriever is the first code to call the real embedding model, so these
tests load MiniLM once and query the actual `data/chroma` store — retrieval
must be verified against the corpus it will serve, not a stub.

The two Phase 3 "constraints" are pinned here as concrete cases:
  * the Small Cap exit load question must NOT return the Large Cap chunk first;
  * the Balanced Advantage risk question needs its answer at rank 5, not top-1.
"""

from __future__ import annotations

import pytest

import config
from rag import embedder, store
from rag.retriever import RetrievalResult, detect_scheme, retrieve

CHUNK_COUNT = 31


# --- scheme detection (pure logic, no model) ----------------------------------

@pytest.mark.parametrize("question,code", [
    ("What is the expense ratio of the HDFC Large Cap Fund?", "119018"),
    ("What is the lock-in period for HDFC ELSS Tax Saver Fund?", "119060"),
    ("What is the minimum SIP amount for HDFC Balanced Advantage Fund?", "118968"),
    ("What is the exit load on HDFC Small Cap Fund?", "130503"),
    ("What's the benchmark of HDFC Equity (Flexi Cap) Fund?", "118955"),
    ("What risk category does HDFC Balanced Advantage Fund fall under?", "118968"),
    ("What is the NAV of HDFC Flexi Cap?", "118955"),
    ("Is the ELSS tax saver's lock-in 3 years?", "119060"),
])
def test_detects_the_named_scheme(question, code):
    result = detect_scheme(question)
    assert result.scheme_code == code
    assert not result.out_of_corpus


def test_no_scheme_mention_stays_unfiltered():
    result = detect_scheme("How do I download my capital gains statement?")
    assert result.scheme_code is None
    assert not result.out_of_corpus


@pytest.mark.parametrize("question", [
    "What is the expense ratio of HDFC Mid-Cap?",
    "What is the exit load on Parag Parag Flexi Cap Fund?",
    "Can I redeem HDFC Nifty 50 Index Fund before 3 years?",
    "Compare HDFC with SBI Small Cap Fund.",
])
def test_foreign_fund_is_out_of_corpus(question):
    result = detect_scheme(question)
    assert result.out_of_corpus, question
    assert result.scheme_code is None


def test_mid_cap_does_not_leak_into_any_scheme():
    """'mid cap' must never be read as small/large cap (F10: no number)."""
    result = detect_scheme("What is the expense ratio of HDFC Mid-Cap?")
    assert result.out_of_corpus


# --- retrieval against the real store ----------------------------------------

def test_store_is_populated_before_retrieval():
    assert config.chroma_store_exists(), "run `python ingest.py` first"
    assert store.count() == CHUNK_COUNT


def test_retrieval_uses_the_same_encoder_as_the_store():
    """C7: a different encoder would make the distances meaningless."""
    collection = store.get_collection().metadata
    assert collection["embed_model"] == config.EMBED_MODEL
    assert embedder.model_name() == config.EMBED_MODEL


def test_expense_ratio_question_retrieves_the_overview_chunk():
    result = retrieve("What is the expense ratio of the HDFC Large Cap Fund?")
    assert result.in_scope
    assert result.scheme_code == "119018"
    assert all(c.scheme_code == "119018" for c in result.chunks)
    assert "expense_ratio" in result.best_chunk.fields


def test_small_cap_exit_load_does_not_rank_large_cap_first():
    """Phase 3 constraint 1: pure cosine put the Large Cap chunk first."""
    result = retrieve("What is the exit load on HDFC Small Cap Fund?")
    assert result.scheme_code == "130503"
    assert all(c.scheme_code == "130503" for c in result.chunks)
    assert result.best_chunk.section == "exit_load"
    assert result.best_chunk.distance < config.DISTANCE_THRESHOLD


def test_balanced_advantage_risk_answer_is_at_rank_5():
    """Phase 3 constraint 2: the fact lives one position from the k=5 edge."""
    result = retrieve("What risk category does HDFC Balanced Advantage Fund fall under?")
    assert result.scheme_code == "118968"
    assert len(result.chunks) == config.TOP_K
    risk = [c for c in result.chunks if "risk_rating" in c.fields]
    assert risk, "the risk chunk must be inside the top-k window"
    assert result.in_scope


def test_retrieved_chunks_are_within_threshold():
    result = retrieve("What's the benchmark of HDFC Equity (Flexi Cap) Fund?")
    assert result.best_distance is not None
    assert result.best_distance <= config.DISTANCE_THRESHOLD
    assert result.in_scope


def test_no_scheme_and_no_answer_is_not_in_scope():
    result = retrieve("How do I download my capital gains statement?")
    assert result.scheme_code is None
    assert not result.in_scope
    assert result.best_distance is not None


@pytest.mark.parametrize("question", [
    "What is the expense ratio of HDFC Mid-Cap?",
    "What is the exit load on Parag Parag Flexi Cap Fund?",
    "Can I redeem HDFC Nifty 50 Index Fund before 3 years?",
])
def test_out_of_corpus_question_returns_no_chunks(question):
    result = retrieve(question)
    assert result.out_of_corpus
    assert result.chunks == []
    assert not result.in_scope


def test_returned_chunks_carry_grounding_metadata():
    result = retrieve("What is the minimum lump sum investment for HDFC Large Cap?")
    for chunk in result.chunks:
        assert chunk.source_url in config.ALLOWED_URLS
        assert chunk.scheme_code in config.ALLOWED_SCHEME_CODES
        assert chunk.text
        assert chunk.retrieved_at


def test_top_k_can_be_raised():
    result = retrieve("What risk category does HDFC Balanced Advantage Fund fall under?", k=6)
    assert len(result.chunks) == 6
    assert all(c.scheme_code == "118968" for c in result.chunks)