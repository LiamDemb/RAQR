"""Tests for RouterDataset with synthetic JSONL data."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from raqr.routers.signal_config import (
    LABEL_MAP,
    PROBE_DIM,
    PROBE_KEYS,
    Q_EMB_DIM,
    Q_FEAT_DIM,
    Q_FEAT_KEYS,
    SignalConfig,
)
from raqr.routers.dataset import RouterDataset


def _make_row(
    label: str = "Dense",
    q_emb_dim: int = Q_EMB_DIM,
) -> dict:
    """Create a synthetic labeled row matching the real JSONL schema."""
    row = {
        "question_id": "test_q1",
        "question": "What is X?",
        "gold_label": label,
        "question_embedding": np.random.randn(q_emb_dim).tolist(),
    }
    for k in Q_FEAT_KEYS:
        row[k] = np.random.randint(0, 10)
    for k in PROBE_KEYS:
        row[k] = float(np.random.rand())
    return row


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def tmp_data(tmp_path):
    """Create a temp JSONL file with 20 rows (15 Dense, 5 Graph)."""
    rows = [_make_row("Dense") for _ in range(15)]
    rows += [_make_row("Graph") for _ in range(5)]
    np.random.shuffle(rows)
    path = tmp_path / "test_data.jsonl"
    _write_jsonl(rows, path)
    return path, rows


class TestRouterDatasetLoading:
    def test_loads_all_rows(self, tmp_data):
        path, rows = tmp_data
        config = SignalConfig(use_q_emb=True, use_q_feat=True, use_probe=True)
        ds = RouterDataset(path, config)
        assert len(ds) == len(rows)

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        config = SignalConfig(use_q_emb=True)
        with pytest.raises(ValueError, match="No valid rows"):
            RouterDataset(path, config)

    def test_invalid_label_skipped(self, tmp_path):
        rows = [_make_row("Dense"), _make_row("Dense")]
        rows[1]["gold_label"] = "Unknown"
        path = tmp_path / "bad_label.jsonl"
        _write_jsonl(rows, path)
        config = SignalConfig(use_q_emb=True)
        ds = RouterDataset(path, config)
        assert len(ds) == 1

    def test_missing_signal_raises(self, tmp_path):
        row = _make_row("Dense")
        del row["probe_max_score"]
        path = tmp_path / "missing_probe.jsonl"
        _write_jsonl([row], path)
        config = SignalConfig(use_probe=True)
        with pytest.raises(KeyError, match="probe_max_score"):
            RouterDataset(path, config)


class TestFeatureDimensions:
    def test_q_emb_only(self, tmp_data):
        path, _ = tmp_data
        config = SignalConfig(use_q_emb=True)
        ds = RouterDataset(path, config)
        x, y = ds[0]
        assert x.shape == (Q_EMB_DIM,)
        assert y.dtype == torch.long

    def test_q_feat_only(self, tmp_data):
        path, _ = tmp_data
        config = SignalConfig(use_q_feat=True)
        ds = RouterDataset(path, config)
        x, _ = ds[0]
        assert x.shape == (Q_FEAT_DIM,)

    def test_probe_only(self, tmp_data):
        path, _ = tmp_data
        config = SignalConfig(use_probe=True)
        ds = RouterDataset(path, config)
        x, _ = ds[0]
        assert x.shape == (PROBE_DIM,)

    def test_all_combined(self, tmp_data):
        path, _ = tmp_data
        config = SignalConfig(use_q_emb=True, use_q_feat=True, use_probe=True)
        ds = RouterDataset(path, config)
        x, _ = ds[0]
        assert x.shape == (Q_EMB_DIM + Q_FEAT_DIM + PROBE_DIM,)

    def test_q_feat_plus_probe(self, tmp_data):
        path, _ = tmp_data
        config = SignalConfig(use_q_feat=True, use_probe=True)
        ds = RouterDataset(path, config)
        x, _ = ds[0]
        assert x.shape == (Q_FEAT_DIM + PROBE_DIM,)


class TestLabelEncoding:
    def test_dense_maps_to_0(self, tmp_path):
        path = tmp_path / "dense.jsonl"
        _write_jsonl([_make_row("Dense")], path)
        ds = RouterDataset(path, SignalConfig(use_q_emb=True))
        _, y = ds[0]
        assert y.item() == LABEL_MAP["Dense"]

    def test_graph_maps_to_1(self, tmp_path):
        path = tmp_path / "graph.jsonl"
        _write_jsonl([_make_row("Graph")], path)
        ds = RouterDataset(path, SignalConfig(use_q_emb=True))
        _, y = ds[0]
        assert y.item() == LABEL_MAP["Graph"]


class TestScalerTransfer:
    def test_dev_uses_train_scaler(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        dev_path = tmp_path / "dev.jsonl"
        _write_jsonl([_make_row("Dense") for _ in range(10)], train_path)
        _write_jsonl([_make_row("Graph") for _ in range(3)], dev_path)

        config = SignalConfig(use_q_feat=True, use_probe=True)
        train_ds = RouterDataset(train_path, config)
        dev_ds = RouterDataset(dev_path, config, scaler=train_ds.scaler)

        assert dev_ds.scaler is train_ds.scaler
        assert len(dev_ds) == 3
