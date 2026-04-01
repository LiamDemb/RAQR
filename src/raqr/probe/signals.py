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
    entropy: float
    gini_softmax: float
    mass_k_80: float
    mass_k_90: float
    mass_k_95: float
    top1_top2_gap: float
    top1_top2_ratio: float
