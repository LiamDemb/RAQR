"""LLM-based relation triple extractor (POC).

Uses OpenAI Chat API with structured output (tool calling) to extract
subject-predicate-object triples from text. Output shape matches REBEL/rule
extractors for easy integration into graph build.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

from raqr.data.enrich_entities import norm_entity
from raqr.prompts import get_triple_extractor_prompt

logger = logging.getLogger(__name__)

RULE_ID = "LLM_TRIPLE_V1"

_EXTRACT_TRIPLES_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_triples",
        "description": "Emit extracted subject-predicate-object triples from the text.",
        "parameters": {
            "type": "object",
            "properties": {
                "triples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subj_surface": {"type": "string", "description": "Subject entity as mentioned"},
                            "pred": {"type": "string", "description": "Relation predicate (snake_case)"},
                            "obj_surface": {"type": "string", "description": "Object entity or value"},
                            "confidence": {"type": "number", "description": "Confidence 0-1"},
                            "evidence": {"type": "string", "description": "Exact quote from text"},
                        },
                        "required": ["subj_surface", "pred", "obj_surface"],
                    },
                },
            },
            "required": ["triples"],
        },
    },
}


def _find_evidence_span(evidence: str, text: str) -> tuple[int, int]:
    """Locate evidence string in text. Returns (start, end) or (-1, -1) if not found."""
    if not evidence or not text:
        return (-1, -1)
    # Try exact match first
    idx = text.find(evidence)
    if idx >= 0:
        return (idx, idx + len(evidence))
    # Try normalized (collapse whitespace)
    ev_norm = " ".join(evidence.split())
    txt_norm = " ".join(text.split())
    idx = txt_norm.find(ev_norm)
    if idx >= 0:
        return (idx, idx + len(ev_norm))
    return (-1, -1)


def _post_process_raw_triples(
    raw_triples: List[Dict[str, Any]],
    text: str,
    alias_map: Dict[str, str],
    chunk_id: Optional[str],
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Convert raw LLM tool-call triples into REBEL-compatible dicts."""
    out: List[Dict[str, Any]] = []
    seen: set = set()

    for t in raw_triples:
        subj_surface = (t.get("subj_surface") or "").strip()
        pred = (t.get("pred") or "").strip()
        obj_surface = (t.get("obj_surface") or "").strip()
        confidence = t.get("confidence")
        if confidence is None:
            confidence = 0.8
        else:
            confidence = max(0.0, min(1.0, float(confidence)))
        evidence = (t.get("evidence") or "").strip()

        if not subj_surface or not pred or not obj_surface:
            if debug:
                logger.debug("Skipping incomplete triple: %s", t)
            continue

        subj_norm = norm_entity(subj_surface, alias_map)
        obj_norm = norm_entity(obj_surface, alias_map)

        if not subj_norm or not obj_norm:
            if debug:
                logger.debug("Skipping triple with empty norm: subj=%r obj=%r", subj_surface, obj_surface)
            continue

        key = (subj_norm, pred, obj_norm)
        if key in seen:
            continue
        seen.add(key)

        start_char, end_char = _find_evidence_span(evidence, text)

        rec: Dict[str, Any] = {
            "subj_surface": subj_surface,
            "obj_surface": obj_surface,
            "subj_norm": subj_norm,
            "pred": pred,
            "obj_norm": obj_norm,
            "source": "llm",
            "rule_id": RULE_ID,
            "confidence": confidence,
            "match_text": evidence or f"{subj_surface} {pred} {obj_surface}",
            "start_char": start_char,
            "end_char": end_char,
        }
        if chunk_id:
            rec["chunk_id"] = chunk_id
        out.append(rec)

    return out


@dataclass
class LLMTripleExtractor:
    """LLM-based triple extractor using OpenAI Chat API with tool calling."""

    model_id: str = ""
    temperature: float = 0.0
    max_tokens: int = 0
    prompt_template: str = ""
    _client: Optional[OpenAI] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.model_id:
            self.model_id = os.getenv("LLM_TRIPLE_MODEL", "gpt-4o-mini")
        if self.temperature == 0.0:
            env_temp = os.getenv("LLM_TRIPLE_TEMPERATURE")
            if env_temp is not None:
                self.temperature = float(env_temp)
        if self.max_tokens <= 0:
            self.max_tokens = int(os.getenv("LLM_TRIPLE_MAX_TOKENS", "2048"))
        if not self.prompt_template:
            self.prompt_template = get_triple_extractor_prompt()

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def _call_llm(self, text: str) -> List[Dict[str, Any]]:
        """Call the LLM and return raw triples from tool-call args. Overridable for testing."""
        prompt = self.prompt_template.format(text=text)
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            tools=[_EXTRACT_TRIPLES_TOOL],
            tool_choice={"type": "function", "function": {"name": "extract_triples"}},
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        raw_triples: List[Dict[str, Any]] = []
        msg = response.choices[0].message if response.choices else None
        if msg and msg.tool_calls:
            for tc in msg.tool_calls:
                if getattr(tc, "function", None) and getattr(tc.function, "name", None) == "extract_triples":
                    args_str = getattr(tc.function, "arguments", None) or "{}"
                    try:
                        args = json.loads(args_str)
                        raw_triples.extend(args.get("triples", []))
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse extract_triples args: %s", e)
        return raw_triples

    def extract(
        self,
        text: str,
        chunk_id: Optional[str] = None,
        alias_map: Optional[Dict[str, str]] = None,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """Extract triples from text. Returns REBEL-compatible dict list."""
        alias_map = alias_map or {}
        text = text or ""
        if not text.strip():
            return []

        raw_triples = self._call_llm(text)
        return _post_process_raw_triples(raw_triples, text, alias_map, chunk_id, debug=debug)


# ---------------------------------------------------------------------------
# Batch API helpers
# ---------------------------------------------------------------------------


def build_chat_completion_request(
    prompt: str,
    model_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Dict[str, Any]:
    """Build the body dict for /v1/chat/completions used by the Batch API."""
    model_id = model_id or os.getenv("LLM_TRIPLE_MODEL", "gpt-4o-mini")
    temperature = temperature if temperature is not None else float(os.getenv("LLM_TRIPLE_TEMPERATURE", "0"))
    max_tokens = max_tokens if max_tokens is not None else int(os.getenv("LLM_TRIPLE_MAX_TOKENS", "2048"))
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [_EXTRACT_TRIPLES_TOOL],
        "tool_choice": {"type": "function", "function": {"name": "extract_triples"}},
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def build_batch_line(custom_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Build a single line for the Batch API input JSONL file."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def parse_batch_output_line(line: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]]]:
    """Parse a line from the Batch API output JSONL into (custom_id, raw_triples).

    Returns empty raw_triples if response is missing, error, or parse fails.
    """
    custom_id = line.get("custom_id") or ""
    raw_triples: List[Dict[str, Any]] = []

    response = line.get("response")
    error = line.get("error")
    if error:
        logger.debug("Batch output line %s has error: %s", custom_id, error)
        return (custom_id, [])

    if not response or response.get("status_code") != 200:
        return (custom_id, [])

    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return (custom_id, [])

    msg = choices[0].get("message") or {}
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        func = tc.get("function") or {}
        if func.get("name") != "extract_triples":
            continue
        args_str = func.get("arguments") or "{}"
        try:
            args = json.loads(args_str)
            raw_triples.extend(args.get("triples", []))
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse extract_triples args for %s: %s", custom_id, e)
    return (custom_id, raw_triples)
