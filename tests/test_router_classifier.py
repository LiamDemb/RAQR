"""Tests for RouterClassifier (XGBoost) shapes and predict consistency."""

from __future__ import annotations

import numpy as np
import pytest

from raqr.routers.classifier import RouterClassifier
from raqr.routers.signal_config import SignalConfig

ALL_CONFIGS = [
    SignalConfig(use_q_emb=True),
    SignalConfig(use_q_feat=True),
    SignalConfig(use_probe=True),
    SignalConfig(use_q_emb=True, use_q_feat=True),
    SignalConfig(use_q_emb=True, use_probe=True),
    SignalConfig(use_q_feat=True, use_probe=True),
    SignalConfig(use_q_emb=True, use_q_feat=True, use_probe=True),
]


class TestPredictProba:
    @pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.identifier)
    def test_output_shape_after_fit(self, config: SignalConfig) -> None:
        n, d = 24, config.input_dim
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, d)).astype(np.float32)
        y = rng.integers(0, 2, size=n, dtype=np.int64)
        model = RouterClassifier(
            config=config, n_estimators=12, max_depth=3, n_jobs=1, random_state=0
        )
        model.fit(X, y, verbose=False)
        proba = model.predict_proba(X)
        assert proba.shape == (n, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)

    def test_single_row(self) -> None:
        config = SignalConfig(use_q_emb=True)
        rng = np.random.default_rng(1)
        X = rng.standard_normal((1, config.input_dim)).astype(np.float32)
        y = np.array([0], dtype=np.int64)
        model = RouterClassifier(
            config=config, n_estimators=8, max_depth=2, n_jobs=1, random_state=1
        )
        model.fit(X, y, verbose=False)
        out = model.predict_proba(X)
        assert out.shape == (1, 2)


class TestPredictConsistency:
    def test_predict_matches_argmax_proba(self) -> None:
        config = SignalConfig(use_q_feat=True, use_probe=True)
        rng = np.random.default_rng(2)
        X = rng.standard_normal((32, config.input_dim)).astype(np.float32)
        y = rng.integers(0, 2, size=32, dtype=np.int64)
        model = RouterClassifier(
            config=config, n_estimators=20, max_depth=4, n_jobs=1, random_state=2
        )
        model.fit(X, y, verbose=False)
        p_cls = model.predict(X)
        p_arg = np.argmax(model.predict_proba(X), axis=1)
        np.testing.assert_array_equal(p_cls.astype(np.int64), p_arg.astype(np.int64))
