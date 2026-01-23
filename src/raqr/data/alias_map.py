from __future__ import annotations

from typing import Dict, Iterable, List

from .enrich_entities import normalize_key
from .wikipedia_client import WikipediaClient


CURATED_ALIASES = {
    "u.s.": "united states",
    "us": "united states",
    "uk": "united kingdom",
}


def normalize_alias_map(alias_map: Dict[str, str]) -> Dict[str, str]:
    return {normalize_key(k): normalize_key(v) for k, v in alias_map.items()}


def build_alias_map_from_redirects(
    titles: Iterable[str],
    wiki: WikipediaClient,
    batch_size: int = 20,
    curated_aliases: Dict[str, str] | None = None,
) -> Dict[str, str]:
    title_list = [t for t in titles if t]
    redirect_map: Dict[str, List[str]] = {}
    for i in range(0, len(title_list), batch_size):
        batch = title_list[i : i + batch_size]
        redirect_map.update(wiki.fetch_redirects(batch))

    alias_map: Dict[str, str] = {}
    for canonical, redirects in redirect_map.items():
        for redirect in redirects:
            alias_map[normalize_key(redirect)] = normalize_key(canonical)

    curated = curated_aliases or CURATED_ALIASES
    alias_map.update(normalize_alias_map(curated))
    return alias_map
