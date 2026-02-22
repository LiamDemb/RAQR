from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, List, Optional

import spacy

DEFAULT_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART"}
DEFAULT_SPACY_MODEL = "en_core_web_sm"


def normalize_key(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).lower()
    s = re.sub(r"['’]s\b", "", s)
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Keep alias/entity lookup deterministic across common title forms.
    while True:
        stripped = re.sub(r"^(the|a|an)\s+", "", s).strip()
        if stripped == s:
            break
        s = stripped
    return s


def norm_entity(text: str, alias_map: Optional[Dict[str, str]] = None) -> str:
    alias_map = alias_map or {}
    normalized = normalize_key(text)
    return alias_map.get(normalized, normalized)


def load_spacy(model: str | None = None) -> "spacy.Language":
    model = model or os.environ.get("SPACY_MODEL", DEFAULT_SPACY_MODEL)
    try:
        return spacy.load(model, disable=["tagger", "parser", "lemmatizer"])
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{model}' not found. Install it with:\n"
            f"  python -m spacy download {model}\n"
            "Or run: make setup-models"
        ) from e


def extract_entities_spacy(
    text: str,
    nlp,
    alias_map: Optional[Dict[str, str]] = None,
    allowed_types: Optional[set] = None,
) -> List[Dict[str, str]]:
    allowed = allowed_types or DEFAULT_ENTITY_TYPES
    alias_map = alias_map or {}
    ents: Dict[str, Dict[str, str]] = {}
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ not in allowed:
            continue
        norm = norm_entity(ent.text, alias_map)
        if not norm:
            continue
        ents[norm] = {
            "surface": ent.text,
            "norm": norm,
            "type": ent.label_,
            "qid": None,
        }
    return list(ents.values())
