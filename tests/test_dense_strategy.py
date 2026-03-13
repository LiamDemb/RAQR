"""Tests for the Dense retrieval strategy."""

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import faiss
import numpy as np
import pandas as pd

from raqr.generator import GenerationResult
from raqr.strategies.dense import DenseStrategy
from raqr.strategies.base import StrategyResult


def _make_tiny_index_and_meta(tmp_path: Path, dim: int = 384):
    """Build a minimal FAISS index + metadata for tests."""
    rng = np.random.default_rng(42)
    n = 5
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    idx_path = tmp_path / "vector_index.faiss"
    faiss.write_index(index, str(idx_path))
    meta = pd.DataFrame(
        [
            {"row_id": i, "chunk_id": f"c{i}", "year_min": None, "year_max": None}
            for i in range(n)
        ]
    )
    meta_path = tmp_path / "vector_meta.parquet"
    meta.to_parquet(meta_path, index=False)
    return str(idx_path), str(meta_path)


class MockCorpus:
    """Minimal ChunkIdToText implementation for tests."""

    def __init__(self, texts: dict[str, str]):
        self._texts = texts

    def get_text(self, chunk_id: str) -> Optional[str]:
        return self._texts.get(chunk_id)


def test_strategy_returns_valid_strategy_result(tmp_path):
    """DenseStrategy returns a StrategyResult with correct shape and latency keys."""
    idx_path, meta_path = _make_tiny_index_and_meta(tmp_path, dim=384)

    # Mock embedder returns vecs[0] so we retrieve row 0
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((5, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    query_vec = vecs[0]

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = query_vec

    mock_generator = MagicMock()
    mock_generator.generate.return_value = GenerationResult(
        text="Paris is the capital of France.",
        model_id="test",
        latency_ms=10.0,
        prompt_hash="abc",
        sampling={},
    )

    corpus = MockCorpus({f"c{i}": f"text for chunk {i}" for i in range(5)})

    from raqr.index_store import FaissIndexStore
    from raqr.loaders import VectorMetaMapper

    strategy = DenseStrategy(
        index_store=FaissIndexStore(index_path=idx_path),
        meta=VectorMetaMapper(parquet_path=meta_path),
        embedder=mock_embedder,
        generator=mock_generator,
        corpus=corpus,
        top_k=3,
    )

    result = strategy.retrieve_and_generate("What is the capital of France?")

    assert isinstance(result, StrategyResult)
    assert result.status == "OK"
    assert result.answer == "Paris is the capital of France."
    assert len(result.context_scores) > 0
    assert len(result.context_scores) <= 3
    assert "retrieval" in result.latency_ms
    assert "generation" in result.latency_ms
    assert "total" in result.latency_ms
    assert result.latency_ms["total"] >= result.latency_ms["retrieval"]
    assert result.latency_ms["total"] >= result.latency_ms["generation"]
    assert result.error is None


def test_context_scores_sorted_descending(tmp_path):
    """context_scores are sorted by score in descending order."""
    idx_path, meta_path = _make_tiny_index_and_meta(tmp_path, dim=384)

    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((5, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    query_vec = vecs[0]

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = query_vec

    mock_generator = MagicMock()
    mock_generator.generate.return_value = GenerationResult(
        text="Answer", model_id="test", latency_ms=0.0, prompt_hash="", sampling={}
    )

    corpus = MockCorpus({f"c{i}": f"chunk {i}" for i in range(5)})

    from raqr.index_store import FaissIndexStore
    from raqr.loaders import VectorMetaMapper

    strategy = DenseStrategy(
        index_store=FaissIndexStore(index_path=idx_path),
        meta=VectorMetaMapper(parquet_path=meta_path),
        embedder=mock_embedder,
        generator=mock_generator,
        corpus=corpus,
        top_k=5,
    )

    result = strategy.retrieve_and_generate("test query")
    scores = [s for (_, s) in result.context_scores]
    assert scores == sorted(scores, reverse=True)


def test_no_context_when_corpus_returns_empty(tmp_path):
    """When corpus returns no valid chunk texts, status is NO_CONTEXT (chunks filtered out)."""
    idx_path, meta_path = _make_tiny_index_and_meta(tmp_path, dim=384)

    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((5, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    query_vec = vecs[0]

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = query_vec

    # Corpus returns None for all chunks - so pairs will be empty
    mock_corpus = MagicMock()
    mock_corpus.get_text.return_value = None

    from raqr.index_store import FaissIndexStore
    from raqr.loaders import VectorMetaMapper

    strategy = DenseStrategy(
        index_store=FaissIndexStore(index_path=idx_path),
        meta=VectorMetaMapper(parquet_path=meta_path),
        embedder=mock_embedder,
        generator=MagicMock(),
        corpus=mock_corpus,
        top_k=5,
    )

    result = strategy.retrieve_and_generate("test query")
    assert result.status == "NO_CONTEXT"
    assert result.context_scores == []
    assert result.answer == ""
    assert result.error is None
