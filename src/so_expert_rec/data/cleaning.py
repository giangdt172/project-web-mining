"""Text cleaning, dual text views, and tag parsing/canonicalization (Sections 2.5, 3)."""

from __future__ import annotations

import html
import re

_ANGLE_TAG_RE = re.compile(r"<([^<>]+)>")
_PRE_BLOCK_RE = re.compile(r"<pre[^>]*>(.*?)</pre>", flags=re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def parse_so_tags(
    raw_tags: str | None,
    synonym_map: dict[str, str] | None = None,
) -> list[str]:
    """Parse Stack Overflow tags in either `<tag1><tag2>` or `|tag1|tag2|` format.

    Optionally maps synonyms to canonical master tags and deduplicates while preserving order.
    """
    if not raw_tags:
        return []

    raw_tags = raw_tags.strip()
    if "<" in raw_tags and ">" in raw_tags:
        parsed = _ANGLE_TAG_RE.findall(raw_tags)
    elif "|" in raw_tags:
        parsed = [part for part in raw_tags.split("|") if part]
    else:
        parsed = raw_tags.split()

    seen: set[str] = set()
    canonical: list[str] = []
    for tag in parsed:
        t = html.unescape(tag).strip().lower()
        if not t:
            continue
        if synonym_map:
            t = synonym_map.get(t, t)
        if t not in seen:
            seen.add(t)
            canonical.append(t)
    return canonical


def _truncate_code_block(match: re.Match[str], max_code_lines: int) -> str:
    inner = _HTML_TAG_RE.sub("", match.group(1))
    inner = html.unescape(inner)
    lines = [line.strip() for line in inner.splitlines() if line.strip()]
    kept = lines[:max_code_lines]
    return " " + " ".join(kept) + " "


def strip_html_for_bm25(title: str | None, body_html: str | None, max_code_lines: int = 3) -> str:
    """Build the BM25 text view (Section 2.5).

    Keeps title, prose, inline code/identifiers, and the first `max_code_lines` non-empty
    lines of each `<pre>` code block or traceback.
    """
    title_text = html.unescape(title or "").strip()
    body = body_html or ""
    body = _PRE_BLOCK_RE.sub(lambda m: _truncate_code_block(m, max_code_lines), body)
    body = _HTML_TAG_RE.sub(" ", body)
    body = html.unescape(body)
    combined = f"{title_text}\n{body}" if title_text else body
    return _WHITESPACE_RE.sub(" ", combined).strip()


def strip_html_for_dense(title: str | None, body_html: str | None, max_prose_words: int = 200) -> str:
    """Build the dense embedding text view (Section 2.5).

    Removes `<pre>` code blocks, strips HTML tags, and concatenates title + up to
    `max_prose_words` words of cleaned prose.
    """
    title_text = _WHITESPACE_RE.sub(" ", html.unescape(title or "")).strip()
    body = _PRE_BLOCK_RE.sub(" ", body_html or "")
    body = _HTML_TAG_RE.sub(" ", body)
    body = html.unescape(body)
    prose_words = body.split()[:max_prose_words]
    prose = " ".join(prose_words)
    if title_text and prose:
        return f"{title_text}. {prose}"
    return title_text or prose
