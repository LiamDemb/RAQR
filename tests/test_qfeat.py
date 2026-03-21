"""Tests for Q-feat (per-query features).

Keyword-flag tests run without spaCy. Entity count, syntactic depth, and
query length tests require en_core_web_sm and are skipped if missing.
"""

from __future__ import annotations

import pytest

from raqr.features import (
    compute_qfeat,
    get_relational_keywords,
    relational_keyword_flag,
)


class TestRelationalKeywordFlag:
    """Tests for relational_keyword_flag (no spaCy)."""

    def test_keyword_present_returns_1(self):
        assert relational_keyword_flag("What is the relationship between A and B?") == 1

    def test_keyword_with_punctuation_matches(self):
        assert relational_keyword_flag("Compare these items: X, Y.") == 1

    def test_keyword_absent_returns_0(self):
        assert relational_keyword_flag("What is the capital of France?") == 0

    def test_empty_query_returns_0(self):
        assert relational_keyword_flag("") == 0

    def test_all_keywords_listed(self):
        keywords = get_relational_keywords()
        assert "both" in keywords
        assert "shared" in keywords
        assert "cause" in keywords
        assert "impact" in keywords
        assert "relationship" in keywords
        assert "difference" in keywords
        assert "compare" in keywords


class TestQfeatWithSpacy:
    """Tests that require spaCy (skip if model missing)."""

    @pytest.fixture
    def nlp(self):
        try:
            from raqr.features import get_qfeat_nlp

            return get_qfeat_nlp()
        except RuntimeError as e:
            if "not found" in str(e).lower():
                pytest.skip(
                    "spaCy model (en_core_web_sm) not installed. "
                    "Run: python -m spacy download en_core_web_sm"
                )
            raise

    def test_entity_count(self, nlp):
        feats = compute_qfeat("Barack Obama was president of the United States.", nlp=nlp)
        assert feats["entity_count"] >= 1
        assert isinstance(feats["entity_count"], int)

    def test_syntactic_depth(self, nlp):
        feats = compute_qfeat("What is the capital of France?", nlp=nlp)
        assert feats["syntactic_depth"] >= 0
        assert isinstance(feats["syntactic_depth"], int)

    def test_query_length_tokens(self, nlp):
        feats = compute_qfeat("Hello world.", nlp=nlp)
        assert feats["query_length_tokens"] >= 2
        assert isinstance(feats["query_length_tokens"], int)

    def test_empty_query_zero_features(self, nlp):
        feats = compute_qfeat("", nlp=nlp)
        assert feats["entity_count"] == 0
        assert feats["syntactic_depth"] == 0
        assert feats["query_length_tokens"] == 0
