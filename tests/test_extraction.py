"""Extraction guards: no performance leakage, no cross-scheme facts, no boilerplate."""

from __future__ import annotations

import pytest

import config
from rag.extract import PERFORMANCE_RE, ExtractionError, extract_all, extract_page
from rag.loader import load_all

PAGES = load_all()
RECORDS, DROPPED = extract_all(PAGES)


def test_all_five_pages_extracted():
    assert len(PAGES) == 5


def test_detected_scheme_code_matches_config():
    for page in PAGES:
        assert page.detected_scheme_code == page.expected_scheme_code


def test_no_performance_language_in_any_record():
    offenders = [
        (r.scheme_name, r.field, r.value)
        for r in RECORDS
        if (hit := PERFORMANCE_RE.search(r.value))
        and not (
            r.field == "benchmark"
            and r.value.lower().endswith("index")
            and hit.group(0).lower() == "return"
        )
    ]
    assert offenders == []


def test_no_returns_ranking_or_holdings_sections():
    forbidden = {"returns_and_rankings", "holdings", "fund_manager", "tax", "glossary"}
    assert not {r.section for r in RECORDS} & forbidden


def test_every_record_belongs_to_its_page_scheme():
    by_slug = {p.slug: p for p in PAGES}
    for record in RECORDS:
        expected = next(s for s in config.SOURCES if s["url"] == record.source_url)
        assert record.scheme_code == expected["scheme_code"]
        assert record.scheme_name == expected["scheme_name"]
    assert len(by_slug) == 5


def test_no_nfo_risk_boilerplate():
    for record in RECORDS:
        assert "nfo_risk" not in record.value
        assert "Moderately High" not in record.value


def test_risk_rating_is_very_high_not_the_boilerplate():
    risks = {r.value for r in RECORDS if r.field == "risk_rating"}
    assert risks == {"Very High"}


def test_benchmark_present_for_every_scheme():
    codes = {r.scheme_code for r in RECORDS if r.field == "benchmark"}
    assert codes == config.ALLOWED_SCHEME_CODES


def test_expense_ratio_and_min_sip_present_for_every_scheme():
    for field in ("expense_ratio", "min_sip", "nav", "aum"):
        codes = {r.scheme_code for r in RECORDS if r.field == field}
        assert codes == config.ALLOWED_SCHEME_CODES, field


def test_elss_lock_in_is_three_years():
    lockins = [r for r in RECORDS if r.field == "lock_in"]
    assert len(lockins) == 1
    assert lockins[0].value.startswith("3 years")
    assert lockins[0].scheme_code == "119060"


def test_exit_load_present_for_every_scheme():
    codes = {r.scheme_code for r in RECORDS if r.field == "exit_load"}
    assert codes == config.ALLOWED_SCHEME_CODES


def test_no_placeholder_values_survived():
    for record in RECORDS:
        assert record.value.strip()
        assert record.value.lower() not in {"--", "-", "n/a"}


def test_dropped_records_explain_themselves():
    for record in DROPPED:
        assert record.dropped


def test_every_fact_value_appears_verbatim_on_its_page():
    """Nothing may reach a chunk that is not literally on the source page."""
    import re

    for page in PAGES:
        for record in extract_page(page):
            needle = record.value.split(" as of ")[0]
            assert re.search(re.escape(needle), page.text, re.IGNORECASE), (
                record.scheme_code,
                record.field,
                record.value[:80],
            )


def test_extract_page_raises_on_scheme_mismatch():
    page = PAGES[0]
    original = page.expected_scheme_code
    page.expected_scheme_code = "999999"
    try:
        with pytest.raises(ExtractionError):
            extract_page(page)
    finally:
        page.expected_scheme_code = original
