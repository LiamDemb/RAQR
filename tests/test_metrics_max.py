"""Tests for compute_max_f1 and compute_max_em (max over gold answers)."""

import pytest

from raqr.evaluation.metrics import compute_max_f1, compute_max_em


def test_max_f1_single_gold():
    """Single gold: same as compute_f1."""
    assert compute_max_f1("Paris", ["Paris"]) == 1.0
    # "paris" matches "Paris" but "france" adds a token -> F1 < 1
    assert compute_max_f1("paris", ["Paris"]) == 1.0
    assert compute_max_f1("Berlin", ["Paris"]) == 0.0


def test_max_f1_multiple_golds():
    """Max F1 over multiple golds (best match wins)."""
    assert compute_max_f1("Paris", ["Paris", "London"]) == 1.0
    assert compute_max_f1("London", ["Paris", "London"]) == 1.0
    # pred "Paris" exactly matches gold "Paris" -> F1 1.0
    assert compute_max_f1("Paris", ["Paris", "France"]) == 1.0
    # pred matches "Paris" better than "France"
    golds = ["Paris", "France"]
    f1_paris = compute_max_f1("Paris", golds)
    assert f1_paris == 1.0


def test_max_f1_empty_golds():
    """Empty gold list returns 0."""
    assert compute_max_f1("anything", []) == 0.0


def test_max_em_single_gold():
    """Single gold: same as compute_exact_match."""
    assert compute_max_em("Paris", ["Paris"]) == 1.0
    assert compute_max_em("paris", ["Paris"]) == 1.0
    assert compute_max_em("Berlin", ["Paris"]) == 0.0


def test_max_em_multiple_golds():
    """Max EM over multiple golds (any match gives 1.0)."""
    assert compute_max_em("Paris", ["Paris", "London"]) == 1.0
    assert compute_max_em("London", ["Paris", "London"]) == 1.0
    assert compute_max_em("Berlin", ["Paris", "London"]) == 0.0
    assert compute_max_em("four", ["4", "four"]) == 1.0


def test_max_em_empty_golds():
    """Empty gold list returns 0."""
    assert compute_max_em("anything", []) == 0.0
