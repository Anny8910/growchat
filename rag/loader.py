"""Fetch and cache the five allowlisted source pages."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

import config


class SourceFetchError(RuntimeError):
    def __init__(self, url: str, detail: str) -> None:
        super().__init__(f"Failed to load source: {url}\n  {detail}")
        self.url = url


@dataclass
class RawPage:
    slug: str
    url: str
    scheme_name: str
    category: str
    expected_scheme_code: str
    html: str
    text: str
    fetched_at: str
    from_cache: bool

    @property
    def detected_scheme_code(self) -> str | None:
        counts: dict[str, int] = {}
        for code in re.findall(r'"scheme_code":"(\d+)"', self.html):
            counts[code] = counts.get(code, 0) + 1
        if not counts:
            return None
        return max(counts, key=counts.get)


def _cache_paths(slug: str) -> tuple[Path, Path, Path]:
    return (
        config.RAW_DIR / f"{slug}.html",
        config.RAW_DIR / f"{slug}.txt",
        config.RAW_DIR / f"{slug}.meta.json",
    )


def readable_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


MIN_HTML_BYTES = 20_000


def _fetch(url: str) -> str:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": config.USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
            timeout=config.HTTP_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise SourceFetchError(url, f"network error: {exc}") from exc
    if response.status_code != 200:
        raise SourceFetchError(url, f"HTTP {response.status_code}")
    if len(response.text) < MIN_HTML_BYTES:
        raise SourceFetchError(url, f"response too small ({len(response.text)} bytes) - likely a JS shell")
    return response.text


def load_page(source: dict[str, str], force: bool = False) -> RawPage:
    html_path, text_path, meta_path = _cache_paths(source["slug"])

    if not force and html_path.exists() and meta_path.exists():
        cached = html_path.read_text(encoding="utf-8")
        if len(cached) < MIN_HTML_BYTES:
            raise SourceFetchError(
                source["url"],
                f"cached file {html_path} is {len(cached)} bytes - blank or truncated. "
                f"Delete it and re-run, or pass --force.",
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return RawPage(
            slug=source["slug"],
            url=source["url"],
            scheme_name=source["scheme_name"],
            category=source["category"],
            expected_scheme_code=source["scheme_code"],
            html=cached,
            text=text_path.read_text(encoding="utf-8") if text_path.exists() else "",
            fetched_at=meta["fetched_at"],
            from_cache=True,
        )

    html = _fetch(source["url"])
    text = readable_text(html)
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    text_path.write_text(text, encoding="utf-8")
    meta_path.write_text(
        json.dumps({"url": source["url"], "fetched_at": fetched_at}, indent=2),
        encoding="utf-8",
    )

    return RawPage(
        slug=source["slug"],
        url=source["url"],
        scheme_name=source["scheme_name"],
        category=source["category"],
        expected_scheme_code=source["scheme_code"],
        html=html,
        text=text,
        fetched_at=fetched_at,
        from_cache=False,
    )


def load_all(force: bool = False) -> list[RawPage]:
    return [load_page(source, force=force) for source in config.SOURCES]


def snapshot_date(pages: list[RawPage]) -> str:
    return min(page.fetched_at for page in pages)
