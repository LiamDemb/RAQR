"""Tests for Dense vs Graph correctness disagreement at an F1 threshold."""

from raqr.training.strategy_disagreement import strategy_disagreement


def test_both_above_threshold_not_disagreement():
    assert strategy_disagreement(0.8, 0.9, 0.5) is False


def test_both_below_threshold_not_disagreement():
    assert strategy_disagreement(0.2, 0.1, 0.5) is False


def test_one_above_one_below_is_disagreement():
    assert strategy_disagreement(0.6, 0.2, 0.5) is True
    assert strategy_disagreement(0.1, 0.7, 0.5) is True


def test_boundary_inclusive_correct():
    """F1 >= threshold counts as correct; both at threshold → both correct → no disagreement."""
    assert strategy_disagreement(0.5, 0.5, 0.5) is False
    assert strategy_disagreement(0.5, 0.49, 0.5) is True
