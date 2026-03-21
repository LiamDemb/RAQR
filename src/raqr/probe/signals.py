"""Probe signal dataclass for router input."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeSignals:
    """Signals derived from a top-k probe."""

    max_score: float
    min_score: float
    score_sd: float
    skewness: float
    semantic_dispersion: float
