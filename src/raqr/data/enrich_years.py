from __future__ import annotations

import re
from typing import Dict, List

YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2}|2100)\b")
RANGE_RE = re.compile(r"\b(19\d{2}|20\d{2})\s*(?:–|-|to)\s*(19\d{2}|20\d{2})\b")


def extract_years(text: str, max_range_expand: int = 15) -> List[int]:
    years = set(int(y) for y in YEAR_RE.findall(text))
    for y1, y2 in RANGE_RE.findall(text):
        a, b = int(y1), int(y2)
        if a > b:
            a, b = b, a
        if (b - a) <= max_range_expand:
            years.update(range(a, b + 1))
        else:
            years.update([a, b])
    return sorted(years)


def aggregate_year_fields(years: List[int], text: str, token_count: int) -> Dict[str, float | int | List[int]]:
    if not years:
        return {"years": [], "year_min": None, "year_max": None, "temporal_density": 0.0}
    density = (len(years) / max(token_count, 1)) if token_count else 0.0
    return {
        "years": years,
        "year_min": years[0],
        "year_max": years[-1],
        "temporal_density": density,
    }
