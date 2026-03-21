"""Tests for oracle label determination (δ logic, Dense tie-break)."""

import pytest

from raqr.evaluation.oracle import determine_oracle_label


def test_graph_wins_with_margin():
    """Graph must beat Dense by at least delta."""
    assert determine_oracle_label(0.5, 0.6, delta=0.05) == "Graph"
    assert determine_oracle_label(0.4, 0.5, delta=0.05) == "Graph"
    assert determine_oracle_label(0.0, 0.06, delta=0.05) == "Graph"


def test_dense_wins_tie():
    """Ties go to Dense."""
    assert determine_oracle_label(0.5, 0.5, delta=0.05) == "Dense"
    assert determine_oracle_label(1.0, 1.0, delta=0.05) == "Dense"


def test_dense_wins_near_tie():
    """Near-ties (within delta) go to Dense."""
    assert determine_oracle_label(0.5, 0.52, delta=0.05) == "Dense"
    assert determine_oracle_label(0.5, 0.54, delta=0.05) == "Dense"


def test_graph_exactly_at_margin():
    """Graph at exactly dense + delta wins."""
    assert determine_oracle_label(0.5, 0.55, delta=0.05) == "Graph"


def test_dense_wins_when_better():
    """Dense wins when Dense F1 exceeds Graph F1."""
    assert determine_oracle_label(0.8, 0.5, delta=0.05) == "Dense"
    assert determine_oracle_label(0.6, 0.4, delta=0.05) == "Dense"


def test_custom_delta():
    """Custom delta is respected."""
    assert determine_oracle_label(0.5, 0.52, delta=0.02) == "Graph"
    assert determine_oracle_label(0.5, 0.52, delta=0.05) == "Dense"
    assert determine_oracle_label(0.5, 0.52, delta=0.0) == "Graph"
