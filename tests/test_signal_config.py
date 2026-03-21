"""Tests for SignalConfig."""

import pytest

from raqr.routers.signal_config import Q_EMB_DIM, Q_FEAT_DIM, PROBE_DIM, SignalConfig


class TestSignalConfigValidation:
    def test_no_signals_raises(self):
        with pytest.raises(ValueError, match="At least one signal"):
            SignalConfig()

    def test_single_signal_ok(self):
        SignalConfig(use_q_emb=True)
        SignalConfig(use_q_feat=True)
        SignalConfig(use_probe=True)


class TestIdentifier:
    def test_single_signal(self):
        assert SignalConfig(use_q_emb=True).identifier == "q_emb"
        assert SignalConfig(use_probe=True).identifier == "probe"

    def test_multi_signal_order(self):
        cfg = SignalConfig(use_q_emb=True, use_probe=True)
        assert cfg.identifier == "q_emb-probe"

    def test_all_signals(self):
        cfg = SignalConfig(use_q_emb=True, use_q_feat=True, use_probe=True)
        assert cfg.identifier == "q_emb-q_feat-probe"


class TestInputDim:
    def test_q_emb_only(self):
        assert SignalConfig(use_q_emb=True).input_dim == Q_EMB_DIM

    def test_q_feat_only(self):
        assert SignalConfig(use_q_feat=True).input_dim == Q_FEAT_DIM

    def test_probe_only(self):
        assert SignalConfig(use_probe=True).input_dim == PROBE_DIM

    def test_all_combined(self):
        cfg = SignalConfig(use_q_emb=True, use_q_feat=True, use_probe=True)
        assert cfg.input_dim == Q_EMB_DIM + Q_FEAT_DIM + PROBE_DIM


class TestNumScalarFeatures:
    def test_q_emb_has_no_scalars(self):
        assert SignalConfig(use_q_emb=True).num_scalar_features == 0

    def test_q_feat_plus_probe(self):
        cfg = SignalConfig(use_q_feat=True, use_probe=True)
        assert cfg.num_scalar_features == Q_FEAT_DIM + PROBE_DIM


class TestFromStr:
    def test_single(self):
        cfg = SignalConfig.from_str("q_emb")
        assert cfg.use_q_emb and not cfg.use_q_feat and not cfg.use_probe

    def test_multi(self):
        cfg = SignalConfig.from_str("q_emb,probe")
        assert cfg.use_q_emb and not cfg.use_q_feat and cfg.use_probe

    def test_whitespace_tolerance(self):
        cfg = SignalConfig.from_str(" q_feat , probe ")
        assert cfg.use_q_feat and cfg.use_probe

    def test_all(self):
        cfg = SignalConfig.from_str("q_emb,q_feat,probe")
        assert cfg.use_q_emb and cfg.use_q_feat and cfg.use_probe

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            SignalConfig.from_str("")


class TestFrozen:
    def test_immutable(self):
        cfg = SignalConfig(use_q_emb=True)
        with pytest.raises(AttributeError):
            cfg.use_q_emb = False
