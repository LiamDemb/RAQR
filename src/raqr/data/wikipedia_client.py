from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "RAQR/0.1 (research corpus build)"


@dataclass(frozen=True)
class WikiPage:
    title: str
    page_id: Optional[str]
    revision_id: Optional[str]
    url: Optional[str]
    html: Optional[str]
    outgoing_titles: List[str]


class WikipediaClient:
    def __init__(self, throttle_s: float = 0.1) -> None:
        self.throttle_s = throttle_s
        self._last_call = 0.0

    def _sleep_if_needed(self) -> None:
        delta = time.time() - self._last_call
        if delta < self.throttle_s:
            time.sleep(self.throttle_s - delta)
        self._last_call = time.time()

    def _get(self, params: Dict[str, str]) -> dict:
        self._sleep_if_needed()
        resp = requests.get(
            WIKI_API,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def resolve_title(self, title: str) -> Optional[str]:
        data = self._get(
            {
                "action": "query",
                "format": "json",
                "redirects": "1",
                "titles": title,
            }
        )
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            if "missing" not in page:
                return page.get("title")
        return None

    def fetch_html(self, title: str) -> WikiPage:
        data = self._get(
            {
                "action": "parse",
                "format": "json",
                "page": title,
                "prop": "text|links|revid",
                "redirects": "1",
                "formatversion": "2",
            }
        )
        parse = data.get("parse", {})
        html = parse.get("text")
        revid = parse.get("revid")
        page_id = parse.get("pageid")
        resolved_title = parse.get("title", title)
        outgoing = [
            link.get("title")
            for link in parse.get("links", [])
            if link.get("ns") == 0 and link.get("title")
        ]
        url = f"https://en.wikipedia.org/wiki/{resolved_title.replace(' ', '_')}"
        return WikiPage(
            title=resolved_title,
            page_id=str(page_id) if page_id is not None else None,
            revision_id=str(revid) if revid is not None else None,
            url=url,
            html=html,
            outgoing_titles=outgoing,
        )

    @staticmethod
    def parse_redirects_response(data: dict) -> Dict[str, List[str]]:
        redirect_map: Dict[str, List[str]] = {}
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title")
            if not title:
                continue
            redirects = [r.get("title") for r in page.get("redirects", []) if r.get("title")]
            if redirects:
                redirect_map[title] = redirects
        return redirect_map

    def fetch_redirects(self, titles: List[str]) -> Dict[str, List[str]]:
        if not titles:
            return {}
        data = self._get(
            {
                "action": "query",
                "format": "json",
                "prop": "redirects",
                "rdnamespace": "0",
                "rdlimit": "max",
                "redirects": "1",
                "titles": "|".join(titles),
            }
        )
        return self.parse_redirects_response(data)

    def search_titles(self, query: str, limit: int = 5) -> List[str]:
        data = self._get(
            {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": str(limit),
            }
        )
        return [item.get("title") for item in data.get("query", {}).get("search", [])]
