from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np

from raqr.strategies.base import BaseStrategy, StrategyResult
from raqr.embedder import Embedder
from raqr.index_store import FaissIndexStore
from raqr.loaders import RowIdToChunkId, ChunkIdToText
from raqr.generator import Generator

@dataclass
class DenseStrategy(BaseStrategy):
    name = "Dense"
    index_store: FaissIndexStore = None
    meta: RowIdToChunkId = None
    embedder: Embedder = None
    generator: Generator = None
    corpus: ChunkIdToText = None
    top_k: int = int(os.getenv("DENSE_TOP_K", 5))

    _index: Optional[object] = None

    def _ensure_loaded(self) -> None:
        if self._index is None:
            self._index = self.index_store.load()

    def retrieve_and_generate(self, query: str, **kwargs) -> StrategyResult:
        t0 = time.perf_counter()
        timings: Dict[str, float] = {}
        stage = "retrieval"

        try:
            # Retrieval
            r0 = time.perf_counter()
            self._ensure_loaded()

            query_vector = self.embedder.embed_query(query)
            query_matrix = np.expand_dims(query_vector, axis=0) # (1, d) for faiss

            scores, row_ids = self._index.search(query_matrix, self.top_k)
            scores = scores[0].tolist()
            row_ids = row_ids[0].tolist()

            # Filter invalid rows
            pairs: List[Tuple[str, float]] = []
            for score, row_id in zip(scores, row_ids):
                if row_id is None or int(row_id) < 0:
                    continue
                chunk_id = self.meta.row_to_chunk(int(row_id))
                if not chunk_id:
                    continue
                text = self.corpus.get_text(chunk_id)
                if not text:
                    continue
                pairs.append((text, float(score)))

            # Ensure consistent ordering (descending score)
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

            # Generation
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
            # Failure in generation or retrieval
            timings["total"] = (time.perf_counter() - t0) * 1000.0
            error_msg = (
                f"DenseStrategy failed during {stage}: "
                f"{type(e).__name__}: {e}"
            )
            return StrategyResult(
                answer="",
                context_scores=[],
                latency_ms=timings,
                status="ERROR",
                error=error_msg,
            )