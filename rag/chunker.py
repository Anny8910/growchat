"""Pack fact records into self-describing chunks and dump them for inspection."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import config
from .extract import FactRecord

RULE = "=" * 78
THIN_RULE = "-" * 78


class ChunkError(RuntimeError):
    """The chunk corpus is missing, malformed, or internally inconsistent."""


@dataclass
class Chunk:
    text: str
    source_url: str
    scheme_name: str
    scheme_code: str
    category: str
    section: str
    fields: list[str]
    chunk_index: int
    retrieved_at: str
    content_hash: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)

    def metadata(self) -> dict[str, str | int]:
        return {
            "source_url": self.source_url,
            "scheme_name": self.scheme_name,
            "scheme_code": self.scheme_code,
            "category": self.category,
            "section": self.section,
            "fields": ",".join(self.fields),
            "chunk_index": self.chunk_index,
            "retrieved_at": self.retrieved_at,
            "content_hash": self.content_hash,
        }


def _statement(record: FactRecord, default_as_of: str) -> str:
    if record.field == "objective":
        return f"Investment objective: {record.value}"
    if record.field == "scheme_summary":
        return record.value
    label = record.label
    if record.as_of != default_as_of:
        label = f"{label} (effective {record.as_of})"
    return f"{label}: {record.value}."


def _split_prose(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    step = max_words - overlap
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), step) if i < len(words)]


def _header(record: FactRecord) -> str:
    return f"{record.scheme_name} — {record.category}"


def _hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_chunks(records: list[FactRecord], default_as_of: str) -> list[Chunk]:
    """One scheme and one section per chunk, capped at CHUNK_WORDS."""
    groups: dict[tuple[str, str], list[FactRecord]] = {}
    for record in records:
        groups.setdefault((record.scheme_code, record.section), []).append(record)

    chunks: list[Chunk] = []

    def push(group: list[FactRecord], section: str, body: str, fields: list[str]) -> None:
        head = _header(group[0])
        text = f"{head}\n{body}"
        chunks.append(
            Chunk(
                text=text,
                source_url=group[0].source_url,
                scheme_name=group[0].scheme_name,
                scheme_code=group[0].scheme_code,
                category=group[0].category,
                section=section,
                fields=fields,
                chunk_index=len(chunks),
                retrieved_at=default_as_of,
                content_hash=_hash(text),
            )
        )

    for (scheme_code, section), group in groups.items():
        group = sorted(group, key=lambda r: (r.field != "scheme_summary", r.field))
        head = _header(group[0])
        has_prose = any(r.kind == "prose" for r in group)
        budget = config.CHUNK_WORDS - len(head.split())

        body = ""
        fields: list[str] = []
        carried: list[str] = []

        for record in group:
            statement = _statement(record, default_as_of)

            if record.kind == "prose":
                for part in _split_prose(statement, budget, config.CHUNK_OVERLAP):
                    push(group, section, part, [record.field])
                body, fields = "", []
                continue

            addition = statement if not body else f" {statement}"
            if body and len((body + addition).split()) > budget:
                if has_prose and carried:
                    body = " ".join(carried)
                else:
                    push(group, section, body, fields)
                    body, fields = "", []
                addition = statement
            body += addition
            if record.field not in fields:
                fields.append(record.field)
            if has_prose:
                carried = statement.split()[-config.CHUNK_OVERLAP :]

        if body:
            push(group, section, body, fields)

    for position, chunk in enumerate(chunks):
        chunk.chunk_index = position
    return chunks


def format_chunk_file(chunks: list[Chunk], pages: list[dict[str, str]]) -> str:
    header = [
        RULE,
        "CHUNKS — Mutual Funds Facts-Only RAG Chatbot",
        RULE,
        f"chunks        : {len(chunks)}",
        f"words per chunk cap : {config.CHUNK_WORDS}  (overlap {config.CHUNK_OVERLAP}, prose only)",
        f"snapshot date : {chunks[0].retrieved_at if chunks else 'n/a'}",
        f"sources       : {len(pages)}",
        "",
        "Sources:",
    ]
    for page in pages:
        header.append(f"  [{page['scheme_code']}] {page['scheme_name']}")
        header.append(f"      {page['url']}")
    header.append("")

    blocks = []
    for chunk in chunks:
        blocks.append(
            "\n".join(
                [
                    RULE,
                    f"CHUNK {chunk.chunk_index + 1:03d} of {len(chunks):03d}",
                    RULE,
                    f"source_url   : {chunk.source_url}",
                    f"scheme_name  : {chunk.scheme_name}",
                    f"scheme_code  : {chunk.scheme_code}",
                    f"category     : {chunk.category}",
                    f"section      : {chunk.section}",
                    f"fields       : {', '.join(chunk.fields)}",
                    f"retrieved_at : {chunk.retrieved_at}",
                    f"content_hash : {chunk.content_hash}",
                    f"words        : {chunk.word_count}",
                    f"characters   : {chunk.char_count}",
                    THIN_RULE,
                    chunk.text,
                    "",
                ]
            )
        )
    return "\n".join(header) + "\n" + "\n".join(blocks)


def write_chunk_file(chunks: list[Chunk], pages: list[dict[str, str]], path: Path | None = None) -> Path:
    target = path or config.CHUNKS_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(format_chunk_file(chunks, pages), encoding="utf-8")
    return target


_CHUNK_MARKER = re.compile(r"^CHUNK (\d+) of (\d+)$")
_FIELD_LINE = re.compile(r"^([a-z_]+)\s*:\s*(.*)$")
_FIELDS_WITH_INDEX = {"source_url", "scheme_name", "scheme_code", "category",
                      "section", "fields", "retrieved_at", "content_hash"}


def read_chunks(path: Path | None = None) -> list[Chunk]:
    """Parse the committed chunk corpus back into `Chunk` objects.

    The inverse of `format_chunk_file`, so a deploy can build the vector store
    from the reviewed corpus in `data/chunks/chunks.txt` instead of re-scraping
    the source pages. Keeps the deployed facts identical to the audited ones.

    A block is only treated as finished when a RULE is followed by another
    CHUNK marker, so a line of `=` inside a chunk's text cannot truncate it.
    """
    target = path or config.CHUNKS_FILE
    if not target.exists():
        raise ChunkError(f"no chunk corpus at {target}")

    lines = target.read_text(encoding="utf-8").splitlines()
    chunks: list[Chunk] = []
    fields: dict[str, str] = {}
    body: list[str] = []
    index = -1
    in_body = False

    def flush() -> None:
        nonlocal fields, body, index
        if index < 0:
            return
        missing = _FIELDS_WITH_INDEX - fields.keys()
        if missing:
            raise ChunkError(
                f"chunk {index + 1} is missing field(s): {', '.join(sorted(missing))}"
            )
        text = "\n".join(body).strip()
        if not text:
            raise ChunkError(f"chunk {index + 1} has no text")
        chunks.append(Chunk(
            text=text,
            source_url=fields["source_url"],
            scheme_name=fields["scheme_name"],
            scheme_code=fields["scheme_code"],
            category=fields["category"],
            section=fields["section"],
            fields=[f.strip() for f in fields["fields"].split(",") if f.strip()],
            chunk_index=index,
            retrieved_at=fields["retrieved_at"],
            content_hash=fields["content_hash"],
        ))
        fields, body, index = {}, [], -1

    for position, line in enumerate(lines):
        marker = _CHUNK_MARKER.match(line)
        if marker:
            flush()
            in_body = False
            index = int(marker.group(1)) - 1
            continue
        if index < 0:
            continue
        if line == THIN_RULE:
            in_body = True
            continue
        if in_body:
            if line == RULE and _CHUNK_MARKER.match(lines[position + 1] if position + 1 < len(lines) else ""):
                flush()
                in_body = False
            else:
                body.append(line)
            continue
        parsed = _FIELD_LINE.match(line)
        if parsed and parsed.group(1) in _FIELDS_WITH_INDEX:
            fields[parsed.group(1)] = parsed.group(2).strip()

    flush()
    if not chunks:
        raise ChunkError(f"no chunks found in {target}")
    expected = [c.chunk_index for c in chunks]
    if expected != list(range(len(chunks))):
        raise ChunkError(f"chunk indices are not contiguous from 0: {expected}")
    return chunks
