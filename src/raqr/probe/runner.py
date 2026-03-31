"""Load-once DenseProbeRunner for router dataset building."""

from __future__ import annotations

import faiss
import numpy as np
from scipy.stats import skew
from sentence_transformers import SentenceTransformer
import os

from .dense_probe import (
    DEFAULT_MODEL_NAME,
    _compute_semantic_dispersion,
    _compute_standard_deviation,
)


class DenseProbeRunner:
    """Load FAISS index and SentenceTransformer once; run probe per query."""

    def __init__(
        self,
        index_path: str,
        meta_path: str,
        model_name: str = DEFAULT_MODEL_NAME,
        top_k: int = int(os.getenv("PROBE_TOP_K", 30)),
    ) -> None:
        self.index_path = index_path
        self.meta_path = meta_path
        self.model_name = model_name
        self.top_k = top_k
        self._index = faiss.read_index(index_path)
        self._model = SentenceTransformer(model_name)

    def run(self, query: str) -> dict:
        """Run probe and return scores + derived stats.

        Returns:
            probe_scores: list of top-k similarity scores
            probe_max_score, probe_min_score, probe_score_sd, probe_skewness,
            probe_semantic_dispersion
        """
        q_emb = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        k = min(self.top_k, self._index.ntotal)

        if k <= 0:
            return {
                "probe_scores": [],
                "probe_max_score": 0.0,
                "probe_min_score": 0.0,
                "probe_score_sd": 0.0,
                "probe_skewness": 0.0,
                "probe_semantic_dispersion": float("nan"),
            }

        scores, ids = self._index.search(q_emb, k)
        scores = scores[0]
        ids = ids[0]
        valid = ids >= 0

        if not np.any(valid):
            return {
                "probe_scores": [],
                "probe_max_score": 0.0,
                "probe_min_score": 0.0,
                "probe_score_sd": 0.0,
                "probe_skewness": 0.0,
                "probe_semantic_dispersion": float("nan"),
            }

        scores = scores[valid]
        ids = ids[valid]
        scores_list = [float(s) for s in scores]
        scores_arr = scores.astype(np.float32)

        max_s = float(np.max(scores))
        min_s = float(np.min(scores))
        skew_s = float(skew(scores)) if len(scores) > 1 else 0.0
        disp = _compute_semantic_dispersion(q_emb[0], self._index, ids)
        score_sd = _compute_standard_deviation(scores_arr)

        return {
            "probe_scores": scores_list,
            "probe_max_score": max_s,
            "probe_min_score": min_s,
            "probe_score_sd": score_sd,
            "probe_skewness": skew_s,
            "probe_semantic_dispersion": disp,
        }
