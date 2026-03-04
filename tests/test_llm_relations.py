"""Unit tests for LLM triple extractor post-processing and prompt rendering (no API calls)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from raqr.data.llm_relations import (
    LLMTripleExtractor,
    _find_evidence_span,
    _post_process_raw_triples,
    parse_batch_output_line,
)
from raqr.prompts import get_triple_discovery_prompt, get_triple_validation_prompt


# ---------------------------------------------------------------------------
# Evidence span matching
# ---------------------------------------------------------------------------


def test_find_evidence_span_exact_match() -> None:
    text = "Eva Busch was a German cabaret artist."
    evidence = "cabaret artist"
    start, end = _find_evidence_span(evidence, text)
    assert start == 23
    assert end == 23 + len(evidence)  # 37


def test_find_evidence_span_not_found() -> None:
    text = "Eva Busch was a singer."
    evidence = "cabaret artist"
    start, end = _find_evidence_span(evidence, text)
    assert start == -1
    assert end == -1


def test_find_evidence_span_normalized_whitespace() -> None:
    text = "Eva  Busch   was   a   cabaret   artist."
    evidence = "Eva Busch was a cabaret artist"
    start, end = _find_evidence_span(evidence, text)
    assert start >= 0
    assert end > start


def test_find_evidence_span_empty() -> None:
    assert _find_evidence_span("", "hello") == (-1, -1)
    assert _find_evidence_span("x", "") == (-1, -1)


# ---------------------------------------------------------------------------
# Post-processing: normalization with alias_map
# ---------------------------------------------------------------------------


def test_post_process_normalizes_cabaret_artist_to_cabaret() -> None:
    raw = [
        {"subj_surface": "Eva Busch", "pred": "occupation", "obj_surface": "cabaret artist", "evidence": "cabaret artist"},
    ]
    alias_map = {"cabaret artist": "cabaret"}
    text = "Eva Busch was a German cabaret artist."
    out = _post_process_raw_triples(raw, text, alias_map, chunk_id=None)
    assert len(out) == 1
    assert out[0]["subj_norm"] == "eva busch"
    assert out[0]["obj_norm"] == "cabaret"
    assert out[0]["pred"] == "occupation"
    assert out[0]["source"] == "llm"
    assert out[0]["rule_id"] == "LLM_TRIPLE_V1"


def test_post_process_skips_incomplete_triple() -> None:
    raw = [{"subj_surface": "Eva", "pred": "", "obj_surface": "singer"}]
    out = _post_process_raw_triples(raw, "text", {}, None)
    assert len(out) == 0


def test_post_process_dedupes_by_norm_key() -> None:
    raw = [
        {"subj_surface": "Eva Busch", "pred": "occupation", "obj_surface": "singer"},
        {"subj_surface": "Eva Busch", "pred": "occupation", "obj_surface": "Singer"},
    ]
    text = "Eva Busch was a singer."
    out = _post_process_raw_triples(raw, text, {}, None)
    assert len(out) == 1


def test_post_process_rebel_compatible_shape() -> None:
    raw = [{"subj_surface": "Ra", "pred": "deity_of", "obj_surface": "sun", "evidence": "Ra, the sun god"}]
    text = "In Egyptian mythology, Ra, the sun god was central."
    out = _post_process_raw_triples(raw, text, {}, chunk_id="chunk-1")
    assert len(out) == 1
    rec = out[0]
    assert "subj_surface" in rec
    assert "obj_surface" in rec
    assert "subj_norm" in rec
    assert "pred" in rec
    assert "obj_norm" in rec
    assert "rule_id" in rec
    assert "confidence" in rec
    assert "match_text" in rec
    assert "start_char" in rec
    assert "end_char" in rec
    assert rec["chunk_id"] == "chunk-1"
    assert rec["source"] == "llm"


def test_post_process_evidence_span_populated_when_found() -> None:
    text = "Imhotep was high priest of Ra."
    evidence = "high priest of Ra"
    raw = [{"subj_surface": "Imhotep", "pred": "role", "obj_surface": "Ra", "evidence": evidence}]
    out = _post_process_raw_triples(raw, text, {}, None)
    assert len(out) == 1
    assert out[0]["start_char"] >= 0
    assert out[0]["end_char"] > out[0]["start_char"]


def test_post_process_confidence_clamped() -> None:
    raw = [
        {"subj_surface": "A", "pred": "x", "obj_surface": "B", "confidence": 1.5},
        {"subj_surface": "C", "pred": "y", "obj_surface": "D", "confidence": -0.1},
    ]
    out = _post_process_raw_triples(raw, "text", {}, None)
    assert len(out) == 2
    assert out[0]["confidence"] == 1.0
    assert out[1]["confidence"] == 0.0


# ---------------------------------------------------------------------------
# LLMTripleExtractor with mocked _call_llm
# ---------------------------------------------------------------------------


def test_extractor_uses_mocked_llm_output() -> None:
    mock_raw = [
        {"subj_surface": "Eva Busch", "pred": "occupation", "obj_surface": "cabaret artist", "evidence": "cabaret artist"},
    ]
    text = "Eva Busch was a German cabaret artist."
    alias_map = {"cabaret artist": "cabaret"}

    ext = LLMTripleExtractor()
    with patch.object(ext, "_call_llm", return_value=mock_raw):
        triples = ext.extract(text, alias_map=alias_map)

    assert len(triples) == 1
    assert triples[0]["obj_norm"] == "cabaret"


def test_extractor_empty_text_returns_empty() -> None:
    ext = LLMTripleExtractor()
    with patch.object(ext, "_call_llm", return_value=[{"subj_surface": "X", "pred": "p", "obj_surface": "Y"}]):
        triples = ext.extract("")
    assert triples == []


# ---------------------------------------------------------------------------
# Prompt rendering (Discovery + Validation)
# ---------------------------------------------------------------------------


def test_discovery_prompt_has_placeholders() -> None:
    prompt = get_triple_discovery_prompt()
    assert "{text}" in prompt
    assert "{title}" in prompt


def test_validation_prompt_has_placeholders() -> None:
    prompt = get_triple_validation_prompt()
    assert "{text}" in prompt
    assert "{title}" in prompt
    assert "{candidates_from_stage_1}" in prompt


def test_validation_prompt_renders_candidates() -> None:
    candidates = [{"subj_surface": "A", "pred": "p", "obj_surface": "B"}]
    prompt_template = get_triple_validation_prompt()
    rendered = prompt_template.format(
        title="Page",
        text="Some text.",
        candidates_from_stage_1=json.dumps(candidates, ensure_ascii=False),
    )
    assert "A" in rendered
    assert "p" in rendered
    assert "B" in rendered


# ---------------------------------------------------------------------------
# Batch output parsing
# ---------------------------------------------------------------------------


def test_parse_batch_output_line_extracts_triples() -> None:
    line = {
        "custom_id": "chunk_1",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "extract_triples",
                                        "arguments": json.dumps({"triples": [{"subj_surface": "X", "pred": "p", "obj_surface": "Y"}]}),
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        },
    }
    cid, raw = parse_batch_output_line(line)
    assert cid == "chunk_1"
    assert len(raw) == 1
    assert raw[0]["subj_surface"] == "X"
    assert raw[0]["pred"] == "p"
    assert raw[0]["obj_surface"] == "Y"
