"""Dense probe: top-k retrieval + signal extraction (max, skewness, semantic dispersion)."""

from __future__ import annotations

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

DEFAULT_TOP_K = 30
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


def run_probe(
    query: str,
    index_path: str,
    meta_path: str,
    model_name: str = DEFAULT_MODEL_NAME,
    top_k: int = DEFAULT_TOP_K,
) -> ProbeSignals:
    """Run a dense probe and return ProbeSignals."""
    index = faiss.read_index(index_path)
    meta = pd.read_parquet(meta_path)
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)
    k = min(top_k, index.ntotal)
    if k <= 0:
        return ProbeSignals(
            max_score=0.0,
            min_score=0.0,
            score_sd=0.0,
            skewness=0.0,
            semantic_dispersion=float("nan"),
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
        )
    scores = scores[valid]
    ids = ids[valid]
    max_s = float(np.max(scores))
    min_s = float(np.min(scores))
    skew_s = float(skew(scores)) if len(scores) > 1 else 0.0
    disp = _compute_semantic_dispersion(q_emb[0], index, ids)
    score_sd = _compute_standard_deviation(scores)

    return ProbeSignals(
        max_score=max_s,
        min_score=min_s,
        score_sd=score_sd,
        skewness=skew_s,
        semantic_dispersion=disp,
    )
