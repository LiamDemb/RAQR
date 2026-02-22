"""Tests for upgraded Temporal retrieval strategy behavior."""

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from raqr.generator import GenerationResult
from raqr.loaders import VectorMetaWithYears
from raqr.strategies.temporal import TemporalStrategy


class MockCorpus:
    def __init__(self, texts: dict[str, str]):
        self._texts = texts

    def get_text(self, chunk_id: str) -> Optional[str]:
        return self._texts.get(chunk_id)


class FakeIndex:
    def __init__(self, scores: list[float], row_ids: list[int]):
        self._scores = np.array([scores], dtype=np.float32)
        self._row_ids = np.array([row_ids], dtype=np.int64)

    def search(self, x, k: int):
        # k is ignored in this tiny fake; test controls shape.
        return self._scores, self._row_ids


class FakeIndexStore:
    def __init__(self, index: FakeIndex):
        self._index = index

    def load(self):
        return self._index


def _write_meta(tmp_path: Path, rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    path = tmp_path / "vector_meta.parquet"
    df.to_parquet(path, index=False)
    return str(path)


def test_strict_year_presence_rejects_bounds_only_match(tmp_path):
    """Rows are rejected unless target year explicitly appears in years[] metadata."""
    meta_path = _write_meta(
        tmp_path,
        [
            {"row_id": 0, "chunk_id": "c0", "year_min": 1990, "year_max": 2020, "years": [1990, 2020]},
            {"row_id": 1, "chunk_id": "c1", "year_min": 2017, "year_max": 2017, "years": [2017]},
        ],
    )
    strategy = TemporalStrategy(
        index_store=FakeIndexStore(FakeIndex(scores=[0.95, 0.90], row_ids=[0, 1])),
        meta=VectorMetaWithYears(parquet_path=meta_path),
        embedder=MagicMock(embed_query=MagicMock(return_value=np.zeros(4, dtype=np.float32))),
        generator=MagicMock(
            generate=MagicMock(
                return_value=GenerationResult(
                    text="answer", model_id="test", latency_ms=0.0, prompt_hash="", sampling={}
                )
            )
        ),
        corpus=MockCorpus({"c0": "broad years chunk", "c1": "focused 2017 chunk"}),
        top_k=2,
    )

    result = strategy.retrieve_and_generate("What happened in 2017?")
    assert result.status == "OK"
    assert len(result.context_scores) == 1
    assert "focused 2017 chunk" in result.context_scores[0][0]


def test_temporal_density_prefers_focused_chunk_at_same_semantic(tmp_path):
    """When semantic score ties, focused chunk should rank above broad summary chunk."""
    meta_path = _write_meta(
        tmp_path,
        [
            {"row_id": 0, "chunk_id": "c0", "year_min": 2017, "year_max": 2017, "years": [2017]},
            {"row_id": 1, "chunk_id": "c1", "year_min": 2009, "year_max": 2022, "years": [2009, 2017, 2022]},
        ],
    )
    strategy = TemporalStrategy(
        index_store=FakeIndexStore(FakeIndex(scores=[0.90, 0.90], row_ids=[0, 1])),
        meta=VectorMetaWithYears(parquet_path=meta_path),
        embedder=MagicMock(embed_query=MagicMock(return_value=np.zeros(4, dtype=np.float32))),
        generator=MagicMock(
            generate=MagicMock(
                return_value=GenerationResult(
                    text="answer", model_id="test", latency_ms=0.0, prompt_hash="", sampling={}
                )
            )
        ),
        corpus=MockCorpus({"c0": "focused", "c1": "broad"}),
        top_k=2,
        alpha=0.6,
        beta=0.4,
    )

    result = strategy.retrieve_and_generate("What happened in 2017?")
    assert result.status == "OK"
    assert len(result.context_scores) == 2
    assert "focused" in result.context_scores[0][0]
    assert result.context_scores[0][1] > result.context_scores[1][1]


def test_chronological_ordering_applies_after_hybrid_selection(tmp_path):
    """Selected contexts are returned oldest->newest chronologically."""
    meta_path = _write_meta(
        tmp_path,
        [
            {"row_id": 0, "chunk_id": "c0", "year_min": 2018, "year_max": 2018, "years": [2018]},
            {"row_id": 1, "chunk_id": "c1", "year_min": 2016, "year_max": 2016, "years": [2016]},
            {"row_id": 2, "chunk_id": "c2", "year_min": 2017, "year_max": 2017, "years": [2017]},
        ],
    )
    strategy = TemporalStrategy(
        index_store=FakeIndexStore(FakeIndex(scores=[0.99, 0.95, 0.90], row_ids=[0, 1, 2])),
        meta=VectorMetaWithYears(parquet_path=meta_path),
        embedder=MagicMock(embed_query=MagicMock(return_value=np.zeros(4, dtype=np.float32))),
        generator=MagicMock(
            generate=MagicMock(
                return_value=GenerationResult(
                    text="answer", model_id="test", latency_ms=0.0, prompt_hash="", sampling={}
                )
            )
        ),
        corpus=MockCorpus({"c0": "year 2018", "c1": "year 2016", "c2": "year 2017"}),
        top_k=3,
    )

    result = strategy.retrieve_and_generate("What happened from 2016 to 2018?")
    ordered_contexts = [ctx for (ctx, _) in result.context_scores]
    assert ordered_contexts == ["year 2016", "year 2017", "year 2018"]


def test_no_year_query_returns_no_context(tmp_path):
    """No extracted years in query => Temporal not applicable => NO_CONTEXT."""
    meta_path = _write_meta(
        tmp_path,
        [
            {"row_id": 0, "chunk_id": "c0", "year_min": 2017, "year_max": 2017, "years": [2017]},
        ],
    )
    mock_generator = MagicMock()
    strategy = TemporalStrategy(
        index_store=FakeIndexStore(FakeIndex(scores=[0.9], row_ids=[0])),
        meta=VectorMetaWithYears(parquet_path=meta_path),
        embedder=MagicMock(embed_query=MagicMock(return_value=np.zeros(4, dtype=np.float32))),
        generator=mock_generator,
        corpus=MockCorpus({"c0": "chunk"}),
        top_k=1,
    )

    result = strategy.retrieve_and_generate("What is the capital of France?")
    assert result.status == "NO_CONTEXT"
    assert result.context_scores == []
    assert result.answer == ""
    mock_generator.generate.assert_not_called()
