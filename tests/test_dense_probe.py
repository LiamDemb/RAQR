import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from raqr.probe.dense_probe import (
    _compute_distribution_metrics,
    _compute_semantic_dispersion,
    _compute_standard_deviation,
    _gini_softmax_probabilities,
    _shannon_entropy_natural,
    _smallest_k_for_mass_thresholds,
    _softmax,
    _top1_top2_gap_ratio,
    run_probe,
)
from raqr.probe.signals import ProbeSignals


def test_softmax_uniform_three():
    s = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    p = _softmax(s)
    assert len(p) == 3
    assert abs(float(np.sum(p)) - 1.0) < 1e-6
    assert abs(float(p[0]) - 1.0 / 3.0) < 1e-5


def test_entropy_two_equal_masses():
    p = np.array([0.5, 0.5], dtype=np.float64)
    h = _shannon_entropy_natural(p)
    assert abs(h - math.log(2.0)) < 1e-9


def test_gini_uniform():
    p = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    assert abs(_gini_softmax_probabilities(p)) < 1e-9


def test_mass_k_thresholds():
    # Descending mass [0.5, 0.4, 0.1] -> cum [0.5, 0.9, 1.0] (float-safe)
    p = np.array([0.5, 0.4, 0.1], dtype=np.float64)
    m = _smallest_k_for_mass_thresholds(p)
    assert m[0.8] == 2.0
    assert m[0.9] == 2.0
    assert m[0.95] == 3.0


def test_top1_top2_gap_ratio():
    s = np.array([1.0, 0.5], dtype=np.float32)
    gap, ratio = _top1_top2_gap_ratio(s)
    assert gap == 0.5
    assert abs(ratio - 1.0 / (0.5 + 1e-12)) < 1e-6

    g2, r2 = _top1_top2_gap_ratio(np.array([1.0], dtype=np.float32))
    assert math.isnan(g2) and math.isnan(r2)


def test_distribution_metrics_empty():
    t = _compute_distribution_metrics(np.array([], dtype=np.float32))
    assert all(math.isnan(x) for x in t)


def test_standard_deviation():
    # Known std deviation for [1, 2, 3] is sqrt(2/3) ≈ 0.816496
    scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sd = _compute_standard_deviation(scores)
    assert abs(sd - math.sqrt(2/3)) < 1e-5

    # Single item
    assert _compute_standard_deviation(np.array([1.0], dtype=np.float32)) == 0.0

    # Identical items
    assert _compute_standard_deviation(np.array([5.0, 5.0, 5.0], dtype=np.float32)) == 0.0


def test_semantic_dispersion_valid():
    query_emb = np.array([[1.0, 0.0]], dtype=np.float32)
    
    # Mock index
    mock_index = MagicMock()
    # It reconstructs vector [0.0, 1.0] and [1.0, 1.0]
    mock_index.reconstruct.side_effect = lambda i: np.array([0.0, 1.0], dtype=np.float32) if i == 0 else np.array([1.0, 1.0], dtype=np.float32)

    row_ids = np.array([0, 1], dtype=np.int64)
    
    # centroid = [0.5, 1.0] -> norm centroid = [1/sqrt(5), 2/sqrt(5)] ≈ [0.447, 0.894]
    # query_norm = [1.0, 0.0]
    # dot(query, centroid) = 1/sqrt(5) ≈ 0.447213
    # dispersion = 1 - 0.447213 = 0.552786
    expected_disp = 1.0 - (1.0 / math.sqrt(5))
    
    disp = _compute_semantic_dispersion(query_emb, mock_index, row_ids)
    assert abs(disp - expected_disp) < 1e-5


def test_semantic_dispersion_empty():
    disp = _compute_semantic_dispersion(np.array([[1.0, 0.0]], dtype=np.float32), MagicMock(), np.array([], dtype=np.int64))
    assert math.isnan(disp)


def test_semantic_dispersion_zero_norm():
    # Centroid of zero vectors
    mock_index = MagicMock()
    mock_index.reconstruct.side_effect = lambda i: np.array([0.0, 0.0], dtype=np.float32)
    disp = _compute_semantic_dispersion(np.array([[1.0, 0.0]], dtype=np.float32), mock_index, np.array([0, 1], dtype=np.int64))
    assert math.isnan(disp)


@patch("raqr.probe.dense_probe.faiss.read_index")
@patch("raqr.probe.dense_probe.pd.read_parquet")
@patch("raqr.probe.dense_probe.SentenceTransformer")
def test_run_probe(mock_st, mock_read_parquet, mock_read_index):
    # Setup mocks
    mock_index = MagicMock()
    mock_index.ntotal = 100
    mock_read_index.return_value = mock_index
    
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.5, 0.5]], dtype=np.float32)
    mock_st.return_value = mock_model
    
    # Setup FAISS search return
    scores = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
    ids = np.array([[0, 1, 2]], dtype=np.int64)
    mock_index.search.return_value = (scores, ids)
    mock_index.reconstruct.side_effect = lambda i: np.array([1.0, 0.0])
    
    result = run_probe("test query", "dummy_index.faiss", "dummy_meta.parquet", top_k=3)
    
    assert isinstance(result, ProbeSignals)
    assert abs(result.max_score - 0.9) < 1e-5
    assert abs(result.min_score - 0.7) < 1e-5
    assert abs(result.score_sd - math.sqrt(2/300)) < 1e-4 # std(0.9, 0.8, 0.7)
    exp_ent, exp_gini, _, _, _, _, _ = _compute_distribution_metrics(
        np.array([0.9, 0.8, 0.7], dtype=np.float32)
    )
    assert abs(result.entropy - exp_ent) < 1e-5
    assert abs(result.gini_softmax - exp_gini) < 1e-5
    assert math.isfinite(result.mass_k_80)
    
    # Assert ST model was called
    mock_model.encode.assert_called_once_with(["test query"], normalize_embeddings=True)


@patch("raqr.probe.dense_probe.faiss.read_index")
@patch("raqr.probe.dense_probe.pd.read_parquet")
@patch("raqr.probe.dense_probe.SentenceTransformer")
def test_run_probe_invalid_ids(mock_st, mock_read_parquet, mock_read_index):
    # Setup mocks
    mock_index = MagicMock()
    mock_index.ntotal = 100
    mock_read_index.return_value = mock_index
    
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.5, 0.5]], dtype=np.float32)
    mock_st.return_value = mock_model
    
    # FAISS returns -1 ids when not finding enough docs
    scores = np.array([[0.0, 0.0]], dtype=np.float32)
    ids = np.array([[-1, -1]], dtype=np.int64)
    mock_index.search.return_value = (scores, ids)
    
    result = run_probe("test query", "dummy_index.faiss", "dummy_meta.parquet", top_k=3)
    
    assert isinstance(result, ProbeSignals)
    assert result.max_score == 0.0
    assert result.skewness == 0.0
    assert math.isnan(result.semantic_dispersion)
    assert math.isnan(result.entropy)
    assert math.isnan(result.top1_top2_gap)
