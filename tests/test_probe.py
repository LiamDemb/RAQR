"""Tests for the probe module: dispersion computation and reconstruction fallback."""

from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from raqr.probe.dense_probe import _compute_semantic_dispersion, run_probe
from raqr.probe.signals import ProbeSignals


def _make_tiny_index_and_meta(tmpdir: Path, dim: int = 384):
    """Build a minimal FAISS index + metadata for tests."""
    rng = np.random.default_rng(42)
    n = 5
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    idx_path = tmpdir / "vector_index.faiss"
    faiss.write_index(index, str(idx_path))
    meta = pd.DataFrame(
        [{"row_id": i, "chunk_id": f"c{i}", "year_min": None, "year_max": None} for i in range(n)]
    )
    meta_path = tmpdir / "vector_meta.parquet"
    meta.to_parquet(meta_path, index=False)
    return str(idx_path), str(meta_path)


def test_semantic_dispersion_finite_when_reconstruction_works():
    """Semantic dispersion is finite when FAISS index supports reconstruct."""
    rng = np.random.default_rng(42)
    dim = 4
    index = faiss.IndexFlatIP(dim)
    vecs = rng.standard_normal((5, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index.add(vecs)
    q = vecs[0].copy()
    q /= np.linalg.norm(q)
    row_ids = np.array([0, 1, 2])
    disp = _compute_semantic_dispersion(q, index, row_ids)
    assert np.isfinite(disp)
    assert 0 <= disp <= 2


def test_semantic_dispersion_nan_when_reconstruction_fails(caplog):
    """When reconstruct raises, semantic_dispersion is NaN and an error is logged."""
    rng = np.random.default_rng(42)
    index = faiss.IndexFlatIP(4)
    vecs = rng.standard_normal((3, 4)).astype(np.float32)
    index.add(vecs)
    q = np.ones(4, dtype=np.float32) / 2
    row_ids = np.array([0, 1])

    with patch.object(index, "reconstruct", side_effect=RuntimeError("no reconstruct")):
        disp = _compute_semantic_dispersion(q, index, row_ids)

    assert np.isnan(disp)
    assert "does not support reconstruction" in caplog.text or "no reconstruct" in caplog.text


def test_run_probe_returns_probe_signals(tmp_path):
    """run_probe returns ProbeSignals with finite dispersion on IndexFlatIP."""
    idx_path, meta_path = _make_tiny_index_and_meta(tmp_path, dim=384)
    signals = run_probe(
        "What is the capital of France?",
        index_path=idx_path,
        meta_path=meta_path,
        model_name="all-MiniLM-L6-v2",
        top_k=3,
    )
    assert isinstance(signals, ProbeSignals)
    assert signals.max_score >= signals.min_score
    assert np.isfinite(signals.skewness)
    assert np.isfinite(signals.semantic_dispersion)
    assert np.isfinite(signals.entropy)
    assert np.isfinite(signals.gini_softmax)
    assert np.isfinite(signals.mass_k_80)
