"""
Shared helpers for dataset download scripts: Wikipedia URL building and existence checks.

HEAD mode: exact /wiki/Title URL must return HTTP 200 with redirects disabled.
"""

from __future__ import annotations

import email.utils
import html
import os
import random
import re
import time
from collections.abc import Iterable
from urllib.parse import quote

import requests

# Match Wikipedia’s URL rules: spaces → underscores; keep namespace colons etc.
# https://www.mediawiki.org/wiki/Manual:PAGENAMEE_encoding
_WIKI_PATH_SAFE = "/():%!"

_MIN_INTERVAL_S = float(os.environ.get("WIKIPEDIA_MIN_INTERVAL_S", "0.75"))
_MAX_BACKOFF_S = float(os.environ.get("WIKIPEDIA_MAX_BACKOFF_S", "90"))
_MAX_RETRIES = int(os.environ.get("WIKIPEDIA_MAX_RETRIES", "8"))

_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": "RAQRWikipediaLookup/1.0 (research; contact via project maintainer)",
        "Accept-Encoding": "gzip",
    }
)
_last_mono = 0.0


def wiki_article_url(title: str, language: str = "en") -> str:
    """Build /wiki/ URL with UTF-8 percent-encoding."""
    fragment = title.strip().replace(" ", "_")
    encoded = quote(fragment, safe=_WIKI_PATH_SAFE)
    return f"https://{language}.wikipedia.org/wiki/{encoded}"


def _throttle() -> None:
    global _last_mono
    gap = time.monotonic() - _last_mono
    if gap < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - gap)
    _last_mono = time.monotonic()


def _retry_after_seconds(resp: requests.Response) -> float | None:
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(raw)
        if dt is not None:
            return max(0.0, dt.timestamp() - time.time())
    except (TypeError, OSError, ValueError):
        pass
    return None


def _cap_sleep(s: float) -> float:
    return min(max(0.0, s), _MAX_BACKOFF_S)


def wikipedia_find_page_head(title: str, language: str = "en") -> bool:
    """
    True iff HEAD to the exact /wiki/{title} URL returns 200 without following redirects.

    Any 3xx, 404, etc. counts as not found (strict exact URL).
    """
    url = wiki_article_url(title, language=language)

    for attempt in range(_MAX_RETRIES):
        try:
            _throttle()
            response = _SESSION.head(
                url,
                allow_redirects=False,
                timeout=(10, 20),
            )
        except requests.RequestException:
            if attempt == _MAX_RETRIES - 1:
                return False
            time.sleep(_cap_sleep(2.0**attempt + random.random()))
            continue

        if response.status_code == 429:
            ra = _retry_after_seconds(response)
            delay = ra if ra is not None and ra > 0 else 2.0**attempt + random.random()
            time.sleep(_cap_sleep(delay))
            continue

        if response.status_code == 200:
            return True
        return False

    return False


def twowiki_supporting_titles(row: dict) -> set[str]:
    """Unique non-empty titles from 2WikiMultiHopQA supporting_facts."""
    sf = row.get("supporting_facts") or {}
    raw = sf.get("title")
    if raw is None:
        return set()
    if isinstance(raw, str):
        s = raw.strip()
        return {s} if s else set()
    return {str(t).strip() for t in raw if str(t).strip()}


def nq_document_title(row: dict) -> str | None:
    """
    Wikipedia page title for a Natural Questions example.

    NQ stores HTML in document.title; we unescape and strip tags for URL lookup.
    """
    doc = row.get("document")
    title: str | None = None
    if isinstance(doc, dict):
        t = doc.get("title")
        if t is not None and str(t).strip():
            title = str(t).strip()
    if title is None:
        for key in ("document_title", "title"):
            t = row.get(key)
            if t is not None and isinstance(t, str) and t.strip():
                title = t.strip()
                break
    if not title:
        return None

    title = html.unescape(title)
    if "<" in title:
        title = re.sub(r"<[^>]+>", "", title)
    title = " ".join(title.split()).strip()
    return title or None


def titles_all_exist_head(titles: Iterable[str], *, language: str = "en") -> bool:
    """True iff every title passes ``wikipedia_find_page_head``."""
    ts = [t for t in titles if t and str(t).strip()]
    if not ts:
        return False
    for t in ts:
        if not wikipedia_find_page_head(t, language=language):
            return False
    return True
