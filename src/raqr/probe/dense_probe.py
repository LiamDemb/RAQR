"""Dense probe: top-k retrieval + signal extraction (max, skewness, dispersion, softmax stats)."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray
from scipy.stats import skew

from .signals import ProbeSignals

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def _compute_semantic_dispersion(
    query_embedding: np.ndarray,
    index: faiss.Index,
    row_ids: np.ndarray,
) -> float:
    """Compute semantic dispersion via FAISS reconstruction.

    dispersion = 1 - cos(query, centroid(retrieved))
    Requires index to support reconstruct(). If not, returns NaN and logs.
    """

    if len(row_ids) == 0:
        return float("nan")

    try:
        vectors = np.vstack([index.reconstruct(int(i)) for i in row_ids])
    except Exception as e:
        logger.error("Failed to reconstruct vectors for semantic dispersion: %s", e)
        return np.nan

    centroid = vectors.mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(centroid)
    if norm < 1e-10:
        return float("nan")
    centroid_norm = centroid / norm

    q = query_embedding.astype(np.float32).ravel()
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-10:
        return float("nan")
    q = q / q_norm
    sim = float(np.dot(q, centroid_norm))
    return 1.0 - sim


def _compute_standard_deviation(scores: NDArray[np.float32]) -> float:
    mean = float(np.mean(scores))
    sum = 0
    for s in scores:
        sum += (s - mean) ** 2

    return (sum / len(scores)) ** 0.5


def _softmax(scores: NDArray[np.float32]) -> NDArray[np.float64]:
    """Stable softmax over top-k scores; always a valid probability vector."""
    if len(scores) == 0:
        return np.array([], dtype=np.float64)
    x = scores.astype(np.float64)
    x = x - np.max(x)
    exp = np.exp(x)
    s = float(np.sum(exp))
    if not np.isfinite(s) or s <= 0.0:
        n = len(scores)
        return np.full(n, 1.0 / n, dtype=np.float64)
    return exp / s


def _shannon_entropy_natural(p: NDArray[np.float64]) -> float:
    """Shannon entropy with natural log (nats). p must be nonnegative and sum to 1."""
    if len(p) == 0:
        return float("nan")
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _gini_softmax_probabilities(p: NDArray[np.float64]) -> float:
    """Gini coefficient on softmax probabilities, sorted ascending."""
    if len(p) <= 0:
        return float("nan")
    x = np.sort(p.astype(np.float64))
    n = len(x)
    if n == 1:
        return 0.0
    sum_x = float(np.sum(x))
    if sum_x <= 0.0 or not np.isfinite(sum_x):
        return float("nan")
    indices = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.dot(indices, x)) / (n * sum_x) - (n + 1.0) / n)


def _smallest_k_for_mass_thresholds(
    p: NDArray[np.float64],
    thresholds: tuple[float, ...] = (0.8, 0.9, 0.95),
) -> dict[float, float]:
    """Smallest k such that the top-k mass (descending p) reaches each threshold."""
    if len(p) == 0:
        return {t: float("nan") for t in thresholds}
    p_desc = np.sort(p)[::-1]
    cum = np.cumsum(p_desc)
    out: dict[float, float] = {}
    for t in thresholds:
        idx = int(np.searchsorted(cum, t, side="left"))
        if idx >= len(cum):
            out[t] = float(len(cum))
        else:
            out[t] = float(idx + 1)
    return out


def _top1_top2_gap_ratio(
    scores: NDArray[np.float32],
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Raw score gap and ratio; undefined when fewer than two scores (returns NaNs)."""
    if len(scores) < 2:
        return (float("nan"), float("nan"))
    s = np.sort(scores)[::-1]
    s1, s2 = float(s[0]), float(s[1])
    return (s1 - s2, s1 / (s2 + eps))


def _compute_distribution_metrics(
    scores: NDArray[np.float32],
) -> tuple[float, float, float, float, float, float, float]:
    """Entropy (nats), Gini, mass k at 0.8/0.9/0.95, top1-top2 gap and ratio."""
    if len(scores) == 0:
        nan = float("nan")
        return (nan, nan, nan, nan, nan, nan, nan)

    p = _softmax(scores)
    ent = _shannon_entropy_natural(p)
    gini = _gini_softmax_probabilities(p)
    mass = _smallest_k_for_mass_thresholds(p)
    gap, ratio = _top1_top2_gap_ratio(scores)
    return (
        ent,
        gini,
        mass[0.8],
        mass[0.9],
        mass[0.95],
        gap,
        ratio,
    )


def run_probe(
    query: str,
    index_path: str,
    meta_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    top_k: int = int(os.getenv("PROBE_TOP_K", 30)),
) -> ProbeSignals:
    """Run a dense probe and return ProbeSignals."""
    index = faiss.read_index(index_path)
    meta = pd.read_parquet(meta_path)
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    k = min(top_k, index.ntotal)
    nan7 = (float("nan"),) * 7
    if k <= 0:
        return ProbeSignals(
            max_score=0.0,
            min_score=0.0,
            score_sd=0.0,
            skewness=0.0,
            semantic_dispersion=float("nan"),
            entropy=nan7[0],
            gini_softmax=nan7[1],
            mass_k_80=nan7[2],
            mass_k_90=nan7[3],
            mass_k_95=nan7[4],
            top1_top2_gap=nan7[5],
            top1_top2_ratio=nan7[6],
        )
    scores, ids = index.search(q_emb, k)
    scores = scores[0]
    ids = ids[0]
    valid = ids >= 0
    if not np.any(valid):
        return ProbeSignals(
            max_score=0.0,
            min_score=0.0,
            score_sd=0.0,
            skewness=0.0,
            semantic_dispersion=float("nan"),
            entropy=nan7[0],
            gini_softmax=nan7[1],
            mass_k_80=nan7[2],
            mass_k_90=nan7[3],
            mass_k_95=nan7[4],
            top1_top2_gap=nan7[5],
            top1_top2_ratio=nan7[6],
        )
    scores = scores[valid]
    ids = ids[valid]
    max_s = float(np.max(scores))
    min_s = float(np.min(scores))
    skew_s = float(skew(scores)) if len(scores) > 1 else 0.0
    disp = _compute_semantic_dispersion(q_emb[0], index, ids)
    score_sd = _compute_standard_deviation(scores)
    ent, gini, mk80, mk90, mk95, gap, tratio = _compute_distribution_metrics(scores)

    return ProbeSignals(
        max_score=max_s,
        min_score=min_s,
        score_sd=score_sd,
        skewness=skew_s,
        semantic_dispersion=disp,
        entropy=ent,
        gini_softmax=gini,
        mass_k_80=mk80,
        mass_k_90=mk90,
        mass_k_95=mk95,
        top1_top2_gap=gap,
        top1_top2_ratio=tratio,
    )
