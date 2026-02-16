"""Probe signal dataclass for router input."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeSignals:
    """Signals derived from a top-k Dense retrieval probe run.

    Used by the router as dynamic feedback (Probe input channel).
    Canonical term for semantic_dispersion; alias: semantic distance.
    """

    max_score: float
    min_score: float
    mean_score: float
    skewness: float
    semantic_dispersion: float
