from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from raqr.data.enrich_years import extract_years
from raqr.embedder import Embedder
from raqr.generator import Generator
from raqr.index_store import FaissIndexStore
from raqr.loaders import ChunkIdToText
from raqr.strategies.base import BaseStrategy, StrategyResult


def _matched_years(target_years: List[int], chunk_years: List[int]) -> List[int]:
    """Sorted intersection between query years and explicit chunk years metadata."""
    if not target_years or not chunk_years:
        return []
    return sorted(set(target_years).intersection(chunk_years))


@dataclass
class TemporalStrategy(BaseStrategy):
    """Temporal filtering over dense retrieval via explicit year intersections."""

    name = "Temporal"
    index_store: FaissIndexStore = None
    meta: Optional[object] = None  # VectorMetaWithYears: row_to_chunk + get_years
    embedder: Embedder = None
    generator: Generator = None
    corpus: ChunkIdToText = None
    top_k: int = int(os.getenv("TEMPORAL_TOP_K", "5"))
    candidate_multiplier: int = int(os.getenv("TEMPORAL_CANDIDATE_MULTIPLIER", "5"))
    alpha: float = float(os.getenv("TEMPORAL_ALPHA", "0.6"))
    beta: float = float(os.getenv("TEMPORAL_BETA", "0.4"))

    _index: Optional[object] = None
    _last_debug_candidates: List[Dict[str, Any]] = None

    def _ensure_loaded(self) -> None:
        if self._index is None:
            self._index = self.index_store.load()

    def retrieve_and_generate(self, query: str) -> StrategyResult:
        t0 = time.perf_counter()
        timings: Dict[str, float] = {}
        stage = "retrieval"
        self._last_debug_candidates = []

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

            candidates: List[Dict[str, Any]] = []
            for score, row_id in zip(scores, row_ids):
                if row_id is None or int(row_id) < 0:
                    continue
                row_id_int = int(row_id)
                chunk_id = self.meta.row_to_chunk(row_id_int)
                if not chunk_id:
                    continue
                chunk_years = self.meta.get_years(row_id_int)
                if chunk_years is None:
                    continue
                chunk_years = sorted(set(int(y) for y in chunk_years))
                matched = _matched_years(target_years, chunk_years)
                # Strict discard: explicit year must be present in chunk years metadata.
                if not matched:
                    continue
                text = self.corpus.get_text(chunk_id)
                if not text:
                    continue
                temporal_score = len(matched) / max(1, len(chunk_years))
                semantic_score = float(score)
                final_score = (self.alpha * semantic_score) + (self.beta * temporal_score)
                chrono_year = min(matched) if matched else min(chunk_years)
                candidates.append(
                    {
                        "row_id": row_id_int,
                        "chunk_id": chunk_id,
                        "text": text,
                        "semantic_score": semantic_score,
                        "temporal_score": temporal_score,
                        "final_score": final_score,
                        "matched_years": matched,
                        "chunk_years": chunk_years,
                        "chrono_year": chrono_year,
                    }
                )

            # 1) Hybrid rerank
            candidates.sort(key=lambda x: x["final_score"], reverse=True)
            selected = candidates[: self.top_k]
            # 2) Chronological order for prompt + returned context_scores
            selected.sort(key=lambda x: (x["chrono_year"], -x["final_score"]))
            self._last_debug_candidates = selected

            pairs: List[Tuple[str, float]] = [
                (item["text"], float(item["final_score"])) for item in selected
            ]
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
