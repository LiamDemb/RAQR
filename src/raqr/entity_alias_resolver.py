from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, Iterable

import pandas as pd

from raqr.data.alias_map import CURATED_ALIASES, normalize_alias_map
from raqr.data.enrich_entities import normalize_key


def _iter_surface_forms(raw: object) -> Iterable[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, tuple):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (SyntaxError, ValueError):
                return [text]
        return [text]
    return []


@dataclass
class EntityAliasResolver:
    """Resolves entity aliases to canonical normalized keys."""

    alias_map: Dict[str, str]

    @classmethod
    def from_lexicon(
        cls,
        lexicon_path: str,
        norm_col: str = "norm",
        surface_forms_col: str = "surface_forms",
    ) -> "EntityAliasResolver":
        alias_map = normalize_alias_map(CURATED_ALIASES)
        df = pd.read_parquet(lexicon_path, columns=[norm_col, surface_forms_col])
        for _, row in df.iterrows():
            norm_value = normalize_key(str(row[norm_col]))
            if not norm_value:
                continue
            for surface in _iter_surface_forms(row[surface_forms_col]):
                surface_norm = normalize_key(surface)
                if not surface_norm:
                    continue
                alias_map[surface_norm] = norm_value
        return cls(alias_map=alias_map)

    def normalize(self, text: str) -> str:
        key = normalize_key(text)
        return self.alias_map.get(key, key)
