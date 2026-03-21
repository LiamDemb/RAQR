"""Q-feat: Per-query features for dataset formation.

Entity count (spaCy NER), syntactic depth, query length, and relational
keyword flag. Callable per query from your dataset formation script.

Separate from GraphRAG's LLM entity extraction.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from spacy.language import Language

_RELATIONAL_KEYWORDS = frozenset(
    {"both", "shared", "cause", "impact", "relationship", "difference", "compare"}
)

_nlp_cache: "Language | None" = None


def get_qfeat_nlp(model: str | None = None) -> "Language":
    """Load spaCy model for Q-feat (cached singleton).

    Enables parser (dependency) and ner; disables lemmatizer for speed.
    Model: SPACY_MODEL env or en_core_web_sm.
    """
    global _nlp_cache
    if _nlp_cache is not None:
        return _nlp_cache

    import spacy

    model = model or os.environ.get("SPACY_MODEL", "en_core_web_sm")
    try:
        _nlp_cache = spacy.load(model, disable=["lemmatizer"])
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{model}' not found. Install with:\n"
            f"  python -m spacy download {model}\n"
            "Or run: make setup-models"
        ) from e
    return _nlp_cache


def _syntactic_depth(doc) -> int:
    """Maximum depth of dependency parse tree."""
    if not doc or len(doc) == 0:
        return 0

    def depth_to_root(token) -> int:
        d = 0
        t = token
        seen = set()
        while t is not t.head and id(t) not in seen:
            seen.add(id(t))
            d += 1
            t = t.head
        return d

    return max(depth_to_root(tok) for tok in doc) if doc else 0


def _relational_keyword_flag(text: str) -> int:
    """1 if any relational keyword present (case-insensitive), else 0.

    Uses word-boundary extraction so 'relationship,' matches 'relationship'.
    """
    import re

    if not text or not text.strip():
        return 0
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return 1 if (words & _RELATIONAL_KEYWORDS) else 0


def compute_qfeat(
    query: str,
    *,
    nlp: "Language | None" = None,
) -> Dict[str, int | float]:
    """Compute Q-feat for a single query.

    Returns:
        entity_count: Number of spaCy NER entities.
        syntactic_depth: Max depth of dependency parse tree.
        query_length_tokens: Token count.
        relational_keyword_flag: 0 or 1.
    """
    nlp = nlp or get_qfeat_nlp()
    doc = nlp(query) if query and query.strip() else None

    entity_count = len(doc.ents) if doc else 0
    syntactic_depth_val = _syntactic_depth(doc) if doc else 0
    query_length_tokens = len(doc) if doc else 0
    relational_keyword_flag = _relational_keyword_flag(query or "")

    return {
        "entity_count": entity_count,
        "syntactic_depth": syntactic_depth_val,
        "query_length_tokens": query_length_tokens,
        "relational_keyword_flag": relational_keyword_flag,
    }


def relational_keyword_flag(text: str) -> int:
    """1 if any relational keyword present (case-insensitive), else 0.

    No spaCy required. Use for keyword-only checks without loading NLP model.
    """
    return _relational_keyword_flag(text or "")


def get_relational_keywords() -> List[str]:
    """Return the list of relational keywords (for testing or docs)."""
    return sorted(_RELATIONAL_KEYWORDS)
