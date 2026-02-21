from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np

from raqr.data.enrich_years import extract_years
from raqr.embedder import Embedder
from raqr.generator import Generator
from raqr.index_store import FaissIndexStore
from raqr.loaders import ChunkIdToText
from raqr.strategies.base import BaseStrategy, StrategyResult


def _year_bounds_intersect_target(
    year_min: Optional[int],
    year_max: Optional[int],
    target_years: List[int],
) -> bool:
    """True if any target year lies in [year_min, year_max] (inclusive). Missing bounds -> False."""
    if year_min is None or year_max is None or not target_years:
        return False
    for y in target_years:
        if year_min <= y <= year_max:
            return True
    return False


@dataclass
class TemporalStrategy(BaseStrategy):
    """Dense retrieval filtered by query-extracted years; no year signal -> NO_CONTEXT."""

    name = "Temporal"
    index_store: FaissIndexStore = None
    meta: Optional[object] = None  # VectorMetaWithYears: row_to_chunk + get_year_bounds
    embedder: Embedder = None
    generator: Generator = None
    corpus: ChunkIdToText = None
    top_k: int = int(os.getenv("TEMPORAL_TOP_K", "5"))
    candidate_multiplier: int = int(os.getenv("TEMPORAL_CANDIDATE_MULTIPLIER", "5"))

    _index: Optional[object] = None

    def _ensure_loaded(self) -> None:
        if self._index is None:
            self._index = self.index_store.load()

    def retrieve_and_generate(self, query: str) -> StrategyResult:
        t0 = time.perf_counter()
        timings: Dict[str, float] = {}
        stage = "retrieval"

        try:
            target_years = extract_years(query)
            if not target_years:
                timings["total"] = (time.perf_counter() - t0) * 1000.0
                return StrategyResult(
                    answer="",
                    context_scores=[],
                    latency_ms=timings,
                    status="NO_CONTEXT",
                )

            r0 = time.perf_counter()
            self._ensure_loaded()

            query_vector = self.embedder.embed_query(query)
            query_matrix = np.expand_dims(query_vector, axis=0)
            n_candidates = max(self.top_k, self.candidate_multiplier * self.top_k)
            scores, row_ids = self._index.search(query_matrix, n_candidates)
            scores = scores[0].tolist()
            row_ids = row_ids[0].tolist()

            pairs: List[Tuple[str, float]] = []
            for score, row_id in zip(scores, row_ids):
                if len(pairs) >= self.top_k:
                    break
                if row_id is None or int(row_id) < 0:
                    continue
                row_id_int = int(row_id)
                chunk_id = self.meta.row_to_chunk(row_id_int)
                if not chunk_id:
                    continue
                bounds = self.meta.get_year_bounds(row_id_int)
                if bounds is None:
                    continue
                ymin, ymax = bounds
                if not _year_bounds_intersect_target(ymin, ymax, target_years):
                    continue
                text = self.corpus.get_text(chunk_id)
                if not text:
                    continue
                pairs.append((text, float(score)))

            pairs.sort(key=lambda x: x[1], reverse=True)
            timings["retrieval"] = (time.perf_counter() - r0) * 1000.0

            if len(pairs) == 0:
                timings["total"] = (time.perf_counter() - t0) * 1000.0
                return StrategyResult(
                    answer="",
                    context_scores=[],
                    latency_ms=timings,
                    status="NO_CONTEXT",
                )

            context = [t for (t, _) in pairs]
            stage = "generation"
            g0 = time.perf_counter()
            gen = self.generator.generate(query=query, context=context)
            timings["generation"] = (time.perf_counter() - g0) * 1000.0
            timings["total"] = (time.perf_counter() - t0) * 1000.0

            return StrategyResult(
                answer=gen.text,
                context_scores=pairs,
                latency_ms=timings,
                status="OK",
            )

        except Exception as e:
            timings["total"] = (time.perf_counter() - t0) * 1000.0
            error_msg = (
                f"TemporalStrategy failed during {stage}: "
                f"{type(e).__name__}: {e}"
            )
            return StrategyResult(
                answer="",
                context_scores=[],
                latency_ms=timings,
                status="ERROR",
                error=error_msg,
            )
