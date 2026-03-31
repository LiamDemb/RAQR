"""Strategy correctness disagreement (Dense vs Graph) at an F1 threshold."""

from __future__ import annotations


def strategy_disagreement(f1_dense: float, f1_graph: float, threshold: float) -> bool:
    """True iff exactly one strategy is \"correct\" (F1 >= threshold)."""
    d_ok = float(f1_dense) >= threshold
    g_ok = float(f1_graph) >= threshold
    return d_ok != g_ok
