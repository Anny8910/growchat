"""Section-allowlist extraction: page text -> vetted FactRecords.

Allowlist first, denylist second. Only the PRD's in-scope facts are pulled out
(scheme overview, minimum investments, exit load, about, benchmark, ELSS
lock-in). Returns, rankings, holdings, manager and tax prose are never
admitted, and a final performance-term check drops anything that slips past.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field

from .loader import RawPage

PERFORMANCE_TERMS = (
    r"returns?",
    r"annualis",
    r"annualiz",
    r"\bcagr\b",
    r"\brank\b",
    r"category average",
    r"\balpha\b",
    r"historic",
    r"p&l",
    r"would've become",
)

PERFORMANCE_RE = re.compile("|".join(PERFORMANCE_TERMS), re.IGNORECASE)

BENCHMARK_INDEX_RE = re.compile(r"\bIndex$", re.IGNORECASE)
BENCHMARK_EXEMPT_TERM = re.compile(r"^returns?$", re.IGNORECASE)
PLACEHOLDER_VALUES = {"--", "-", "—", "n/a", "na", "not available", "nil return"}

DATE_RE = re.compile(r"^\d{1,2} [A-Z][a-z]{2} \d{4}$")
RISK_RE = re.compile(r"^(Very High|High|Moderately High|Moderate|Moderately Low|Low) Risk$")
LOCKIN_TAG_RE = re.compile(r"^(?P<tag>.+?)\s•\s(?P<years>\d+)Y\s+Lock-in$")
NAV_LABEL_RE = re.compile(r"^NAV:\s*(?P<date>.+?)$", re.IGNORECASE)
EXIT_LOAD_SUMMARY_RE = re.compile(r"^Exit load,? stamp duty and tax$", re.IGNORECASE)

OVERVIEW_LABELS = (
    ("Min. for SIP", "min_sip"),
    ("Fund size (AUM)", "aum"),
    ("Expense ratio", "expense_ratio"),
    ("Rating", "star_rating"),
)

MIN_INVESTMENT_LABELS = (
    ("Min. for 1st investment", "min_first_investment"),
    ("Min. for 2nd investment", "min_second_investment"),
    ("Min. for SIP", "min_sip"),
)

FIELD_LABELS = {
    "expense_ratio": "Expense ratio",
    "min_sip": "Minimum SIP",
    "min_first_investment": "Minimum first investment",
    "min_second_investment": "Minimum second investment",
    "min_lumpsum": "Minimum lump sum investment",
    "exit_load": "Exit load",
    "lock_in": "Lock-in period",
    "risk_rating": "Risk rating",
    "benchmark": "Benchmark",
    "nav": "NAV",
    "aum": "Fund size (AUM)",
    "star_rating": "Star rating",
    "fund_house": "Fund house",
    "launch_date": "Launch date",
}


@dataclass
class FactRecord:
    scheme_code: str
    scheme_name: str
    category: str
    section: str
    field: str
    value: str
    as_of: str
    source_url: str
    kind: str = "fact"
    dropped: list[str] = dc_field(default_factory=list)

    @property
    def label(self) -> str:
        return FIELD_LABELS.get(self.field, self.field.replace("_", " ").capitalize())


class ExtractionError(RuntimeError):
    def __init__(self, page: RawPage, detail: str) -> None:
        super().__init__(f"{page.slug}: {detail}")
        self.page = page


def _clean(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _index_of(lines: list[str], needle: str, start: int = 0) -> int | None:
    for i in range(start, len(lines)):
        if _clean(lines[i]).lower() == needle.lower():
            return i
    return None


def _value_after(lines: list[str], index: int) -> str:
    for line in lines[index + 1 : index + 4]:
        candidate = _clean(line)
        if candidate and candidate not in {"%", "₹", "+", "-"}:
            return candidate
    return ""


def _record(page: RawPage, section: str, name: str, value: str, as_of: str) -> FactRecord:
    return FactRecord(
        scheme_code=page.expected_scheme_code,
        scheme_name=page.scheme_name,
        category=page.category,
        section=section,
        field=name,
        value=_clean(value),
        as_of=as_of,
        source_url=page.url,
    )


def _extract_overview(page: RawPage, lines: list[str]) -> list[FactRecord]:
    risk_idx = next((i for i, l in enumerate(lines) if RISK_RE.match(_clean(l))), None)
    if risk_idx is None:
        raise ExtractionError(page, "risk rating not found; page layout likely changed")

    records: list[FactRecord] = [
        _record(page, "overview", "risk_rating", _clean(lines[risk_idx]).removesuffix(" Risk"), page.fetched_at)
    ]

    for back in range(1, 4):
        tag = LOCKIN_TAG_RE.match(_clean(lines[risk_idx - back]))
        if tag:
            years = int(tag.group("years"))
            records.append(_record(page, "lock_in", "lock_in", f"{years} years", page.fetched_at))
            break

    stop = _index_of(lines, "Return calculator", risk_idx) or len(lines)
    window = lines[risk_idx : min(stop, risk_idx + 40)]

    for i, line in enumerate(window):
        text = _clean(line)
        nav = NAV_LABEL_RE.match(text)
        if nav:
            value = _value_after(window, i)
            if value:
                records.append(
                    _record(page, "overview", "nav", f"{value} as of {nav.group('date').strip()}", page.fetched_at)
                )
            continue
        for label, name in OVERVIEW_LABELS:
            if text == label:
                records.append(_record(page, "overview", name, _value_after(window, i), page.fetched_at))

    seen: set[str] = set()
    return [r for r in records if not (r.field in seen or seen.add(r.field))]


def _extract_minimum_investments(page: RawPage, lines: list[str]) -> list[FactRecord]:
    start = _index_of(lines, "Minimum investments")
    if start is None:
        return []
    stop = _index_of(lines, "Understand terms", start) or min(len(lines), start + 20)
    window = lines[start:stop]

    records: list[FactRecord] = []
    for i, line in enumerate(window):
        text = _clean(line)
        for label, name in MIN_INVESTMENT_LABELS:
            if text == label:
                records.append(_record(page, "minimum_investments", name, _value_after(window, i), page.fetched_at))

    lumpsum = re.search(r"Minimum Lumpsum Investment is\s*(₹[\d,]+)", page.text)
    if lumpsum:
        records.append(
            _record(page, "minimum_investments", "min_lumpsum", lumpsum.group(1), page.fetched_at)
        )
    return records


def _extract_exit_load(page: RawPage, lines: list[str]) -> list[FactRecord]:
    records: list[FactRecord] = []
    start = _index_of(lines, "Exit Load")
    summary_idx = next((i for i, l in enumerate(lines) if EXIT_LOAD_SUMMARY_RE.match(_clean(l))), None)

    if start is not None and summary_idx is not None:
        for i in range(start, summary_idx):
            if DATE_RE.match(_clean(lines[i])):
                effective = _clean(lines[i])
                description = _value_after(lines, i)
                if description and not DATE_RE.match(description) and description.lower() not in PLACEHOLDER_VALUES:
                    records.append(_record(page, "exit_load", "exit_load", description, effective))

    if summary_idx is not None:
        load_idx = _index_of(lines, "Exit load", summary_idx)
        if load_idx is not None and 0 < load_idx - summary_idx <= 3:
            current = _value_after(lines, load_idx)
            if current.lower() not in PLACEHOLDER_VALUES:
                records.insert(0, _record(page, "exit_load", "exit_load", current, page.fetched_at))

    seen: set[str] = set()
    unique: list[FactRecord] = []
    for record in records:
        if record.value in seen:
            continue
        seen.add(record.value)
        unique.append(record)
    return unique


PROSE_DUPLICATE_MARKERS = ("aum", "nav", "minimum", "lumpsum", "exit load", "rated", "stamp duty", "tax")


def _extract_about(page: RawPage, lines: list[str]) -> list[FactRecord]:
    start = _index_of(lines, "About")
    if start is None:
        return []

    benchmark_idx = _index_of(lines, "Fund benchmark", start) or min(len(lines), start + 40)
    obj_idx = next(
        (i for i in range(start, benchmark_idx) if _clean(lines[i]) == "Investment Objective"),
        None,
    )
    summary_end = obj_idx if obj_idx is not None else benchmark_idx

    records: list[FactRecord] = []

    summary = " ".join(
        sentence
        for line in lines[start + 1 : summary_end]
        if _clean(line).endswith((".", "!", "?"))
        for sentence in re.split(r"(?<=[.!?])\s+", _clean(line))
        if len(sentence) > 40
        and not any(marker in sentence.lower() for marker in PROSE_DUPLICATE_MARKERS)
    )
    if summary:
        records.append(_record(page, "about", "scheme_summary", summary, page.fetched_at))
        records[-1].kind = "prose"

    if obj_idx is not None:
        objective = " ".join(
            _clean(line) for line in lines[obj_idx + 1 : benchmark_idx] if len(_clean(line)) > 20
        )
        if objective:
            records.append(_record(page, "about", "objective", objective, page.fetched_at))
            records[-1].kind = "prose"

    return records


def _extract_benchmark(page: RawPage, lines: list[str]) -> list[FactRecord]:
    start = _index_of(lines, "Fund benchmark")
    if start is None:
        return []
    value = _value_after(lines, start)
    return [_record(page, "benchmark", "benchmark", value, page.fetched_at)] if value else []


def _extract_fund_details(page: RawPage, lines: list[str]) -> list[FactRecord]:
    records: list[FactRecord] = []
    for label, name in (("Fund house", "fund_house"), ("Launch Date", "launch_date")):
        start = _index_of(lines, label)
        if start is None:
            continue
        value = _value_after(lines, start)
        if value and not value.startswith(("http", "[", "www")):
            records.append(_record(page, "about", name, value, page.fetched_at))
    return records


def _enforce_guard(page: RawPage, records: list[FactRecord]) -> list[FactRecord]:
    kept: list[FactRecord] = []
    for record in records:
        if not record.value or record.value.lower() in PLACEHOLDER_VALUES:
            record.dropped.append("empty or placeholder value")
            continue
        hit = PERFORMANCE_RE.search(record.value)
        if hit and not _benchmark_exempt(record, hit.group(0)):
            record.dropped.append(f"performance term {hit.group(0)!r} in {record.field}")
            continue
        kept.append(record)
    return kept


def _benchmark_exempt(record: FactRecord, matched_term: str) -> bool:
    """Official index names contain the word 'Return' (NIFTY 500 Total Return Index).

    Only the benchmark field is exempt, and only for the word 'Return' at the
    end of a value the page itself labelled as the fund benchmark.
    """
    return (
        record.field == "benchmark"
        and BENCHMARK_INDEX_RE.search(record.value) is not None
        and BENCHMARK_EXEMPT_TERM.fullmatch(matched_term) is not None
    )


def extract_page(page: RawPage) -> list[FactRecord]:
    """Turn one cached page into vetted fact records."""
    detected = page.detected_scheme_code
    if detected != page.expected_scheme_code:
        raise ExtractionError(
            page,
            f"scheme code mismatch: page says {detected!r}, config says {page.expected_scheme_code!r}",
        )

    lines = page.text.split("\n")
    records: list[FactRecord] = []
    records += _extract_overview(page, lines)
    records += _extract_minimum_investments(page, lines)
    records += _extract_exit_load(page, lines)
    records += _extract_about(page, lines)
    records += _extract_benchmark(page, lines)
    records += _extract_fund_details(page, lines)

    kept = _enforce_guard(page, records)
    if not any(r.section == "overview" for r in kept):
        raise ExtractionError(page, "no overview facts survived extraction")
    return kept


def extract_all(pages: list[RawPage]) -> tuple[list[FactRecord], list[FactRecord]]:
    kept: list[FactRecord] = []
    dropped: list[FactRecord] = []
    for page in pages:
        records = extract_page(page)
        for record in records:
            (dropped if record.dropped else kept).append(record)
    return kept, dropped
