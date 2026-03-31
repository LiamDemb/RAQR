"""Tests for RouterClassifier forward pass shape consistency."""

import pytest
import torch

from raqr.routers.classifier import RouterClassifier, EMB_COMPRESSED_DIM
from raqr.routers.signal_config import (
    PROBE_DIM,
    Q_EMB_DIM,
    Q_FEAT_DIM,
    SignalConfig,
)


ALL_CONFIGS = [
    SignalConfig(use_q_emb=True),
    SignalConfig(use_q_feat=True),
    SignalConfig(use_probe=True),
    SignalConfig(use_q_emb=True, use_q_feat=True),
    SignalConfig(use_q_emb=True, use_probe=True),
    SignalConfig(use_q_feat=True, use_probe=True),
    SignalConfig(use_q_emb=True, use_q_feat=True, use_probe=True),
]


class TestForwardPass:
    @pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.identifier)
    def test_output_shape(self, config: SignalConfig):
        model = RouterClassifier(config=config, num_classes=2)
        batch = torch.randn(8, config.input_dim)
        out = model(batch)
        assert out.shape == (8, 2)

    @pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.identifier)
    def test_single_sample(self, config: SignalConfig):
        model = RouterClassifier(config=config, num_classes=2)
        x = torch.randn(1, config.input_dim)
        out = model(x)
        assert out.shape == (1, 2)

    def test_custom_hidden_dim(self):
        config = SignalConfig(use_q_emb=True)
        model = RouterClassifier(config=config, hidden_dim=64, num_classes=2)
        out = model(torch.randn(4, config.input_dim))
        assert out.shape == (4, 2)


class TestBottleneck:
    def test_emb_compressor_exists_when_q_emb_active(self):
        config = SignalConfig(use_q_emb=True)
        model = RouterClassifier(config=config)
        assert model.emb_compressor is not None
        assert model.emb_compressor.in_features == Q_EMB_DIM
        assert model.emb_compressor.out_features == EMB_COMPRESSED_DIM

    def test_emb_compressor_absent_when_q_emb_inactive(self):
        config = SignalConfig(use_q_feat=True)
        model = RouterClassifier(config=config)
        assert model.emb_compressor is None

    def test_head_input_dim_with_emb(self):
        # head should receive EMB_COMPRESSED_DIM + scalars, not the raw 384
        config = SignalConfig(use_q_emb=True, use_probe=True)
        model = RouterClassifier(config=config)
        expected_head_in = EMB_COMPRESSED_DIM + PROBE_DIM
        assert model.net[0].in_features == expected_head_in

    def test_head_input_dim_scalars_only(self):
        config = SignalConfig(use_q_feat=True, use_probe=True)
        model = RouterClassifier(config=config)
        expected_head_in = Q_FEAT_DIM + PROBE_DIM
        assert model.net[0].in_features == expected_head_in


class TestGradients:
    def test_backward_pass_runs(self):
        config = SignalConfig(use_q_emb=True, use_q_feat=True, use_probe=True)
        model = RouterClassifier(config=config)
        x = torch.randn(4, config.input_dim)
        y = torch.tensor([0, 1, 0, 1])
        loss = torch.nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
