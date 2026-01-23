from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional

import spacy

DEFAULT_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART"}


def normalize_key(text: str) -> str:
    s = unicodedata.normalize("NFKC", text).lower()
    s = re.sub(r"['’]s\b", "", s)
    s = re.sub(r"[^\w\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_entity(text: str, alias_map: Optional[Dict[str, str]] = None) -> str:
    alias_map = alias_map or {}
    normalized = normalize_key(text)
    return alias_map.get(normalized, normalized)


def load_spacy(model: str = "en_core_web_sm"):
    return spacy.load(model, disable=["tagger", "parser", "lemmatizer"])


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
