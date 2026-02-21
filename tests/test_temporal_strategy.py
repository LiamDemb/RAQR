"""Tests for the Temporal retrieval strategy."""

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import faiss
import numpy as np
import pandas as pd

from raqr.generator import GenerationResult
from raqr.loaders import VectorMetaWithYears
from raqr.strategies.base import StrategyResult
from raqr.strategies.temporal import TemporalStrategy


def _make_tiny_index_and_meta_with_years(
    tmp_path: Path,
    dim: int = 384,
    year_spec: Optional[list[tuple[Optional[int], Optional[int]]]] = None,
) -> tuple[str, str]:
    """Build FAISS index + vector_meta.parquet with year_min, year_max per row."""
    if year_spec is None:
        year_spec = [(2016, 2018), (2017, 2017), (2019, 2020), (None, None), (2017, 2018)]
    n = len(year_spec)
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    idx_path = tmp_path / "vector_index.faiss"
    faiss.write_index(index, str(idx_path))
    rows = []
    for i in range(n):
        ymin, ymax = year_spec[i]
        rows.append({
            "row_id": i,
            "chunk_id": f"c{i}",
            "year_min": ymin,
            "year_max": ymax,
        })
    meta = pd.DataFrame(rows)
    meta_path = tmp_path / "vector_meta.parquet"
    meta.to_parquet(meta_path, index=False)
    return str(idx_path), str(meta_path)


class MockCorpus:
    def __init__(self, texts: dict[str, str]):
        self._texts = texts

    def get_text(self, chunk_id: str) -> Optional[str]:
        return self._texts.get(chunk_id)


def test_query_with_year_returns_only_matching_contexts(tmp_path):
    """Query containing a year returns only contexts whose year range contains that year."""
    idx_path, meta_path = _make_tiny_index_and_meta_with_years(tmp_path, dim=384)
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((5, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    query_vec = vecs[0]

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = query_vec
    mock_generator = MagicMock()
    mock_generator.generate.return_value = GenerationResult(
        text="Answer about 2017.",
        model_id="test",
        latency_ms=0.0,
        prompt_hash="",
        sampling={},
    )
    corpus = MockCorpus({f"c{i}": f"chunk {i}" for i in range(5)})

    from raqr.index_store import FaissIndexStore

    strategy = TemporalStrategy(
        index_store=FaissIndexStore(index_path=idx_path),
        meta=VectorMetaWithYears(parquet_path=meta_path),
        embedder=mock_embedder,
        generator=mock_generator,
        corpus=corpus,
        top_k=5,
        candidate_multiplier=2,
    )

    result = strategy.retrieve_and_generate("What happened in 2017?")
    assert result.status == "OK"
    assert len(result.context_scores) > 0
    # Rows with 2017 in range: (2016,2018), (2017,2017), (2017,2018) -> c0, c1, c4
    for ctx, score in result.context_scores:
        assert "chunk 0" in ctx or "chunk 1" in ctx or "chunk 4" in ctx
    assert result.error is None


def test_no_year_in_query_returns_no_context(tmp_path):
    """Query with no detectable year returns NO_CONTEXT."""
    idx_path, meta_path = _make_tiny_index_and_meta_with_years(tmp_path, dim=384)
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = np.zeros(384, dtype=np.float32)
    mock_generator = MagicMock()
    corpus = MockCorpus({f"c{i}": f"chunk {i}" for i in range(5)})

    from raqr.index_store import FaissIndexStore

    strategy = TemporalStrategy(
        index_store=FaissIndexStore(index_path=idx_path),
        meta=VectorMetaWithYears(parquet_path=meta_path),
        embedder=mock_embedder,
        generator=mock_generator,
        corpus=corpus,
        top_k=3,
    )

    result = strategy.retrieve_and_generate("What is the capital of France?")
    assert result.status == "NO_CONTEXT"
    assert result.context_scores == []
    assert result.answer == ""
    assert result.error is None
    mock_generator.generate.assert_not_called()


def test_context_scores_sorted_descending(tmp_path):
    """context_scores are sorted by score in descending order."""
    idx_path, meta_path = _make_tiny_index_and_meta_with_years(tmp_path, dim=384)
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

    strategy = TemporalStrategy(
        index_store=FaissIndexStore(index_path=idx_path),
        meta=VectorMetaWithYears(parquet_path=meta_path),
        embedder=mock_embedder,
        generator=mock_generator,
        corpus=corpus,
        top_k=5,
        candidate_multiplier=2,
    )

    result = strategy.retrieve_and_generate("What happened in 2017 or 2019?")
    scores = [s for (_, s) in result.context_scores]
    assert scores == sorted(scores, reverse=True)


def test_fewer_than_k_matches_returns_fewer(tmp_path):
    """When fewer than top_k chunks match the year filter, return that many (no backfill)."""
    # Only rows 1 and 4 contain 1999; use a tiny spec that only has 1999 in two chunks
    year_spec = [(1998, 2000), (1999, 1999), (2001, 2002), (2003, 2004), (2005, 2006)]
    idx_path, meta_path = _make_tiny_index_and_meta_with_years(tmp_path, dim=384, year_spec=year_spec)
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

    strategy = TemporalStrategy(
        index_store=FaissIndexStore(index_path=idx_path),
        meta=VectorMetaWithYears(parquet_path=meta_path),
        embedder=mock_embedder,
        generator=mock_generator,
        corpus=corpus,
        top_k=5,
        candidate_multiplier=2,
    )

    result = strategy.retrieve_and_generate("What happened in 1999?")
    # Only row 0 (1998-2000) and row 1 (1999) contain 1999
    assert result.status == "OK"
    assert len(result.context_scores) == 2
    assert result.error is None
