"""Tests for RouterClassifier forward pass shape consistency."""

import pytest
import torch

from raqr.routers.classifier import RouterClassifier
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
        model = RouterClassifier(input_dim=config.input_dim, num_classes=2)
        batch = torch.randn(8, config.input_dim)
        out = model(batch)
        assert out.shape == (8, 2)

    @pytest.mark.parametrize("config", ALL_CONFIGS, ids=lambda c: c.identifier)
    def test_single_sample(self, config: SignalConfig):
        model = RouterClassifier(input_dim=config.input_dim, num_classes=2)
        x = torch.randn(1, config.input_dim)
        out = model(x)
        assert out.shape == (1, 2)

    def test_custom_hidden_dim(self):
        model = RouterClassifier(input_dim=Q_EMB_DIM, hidden_dim=64, num_classes=2)
        out = model(torch.randn(4, Q_EMB_DIM))
        assert out.shape == (4, 2)


class TestGradients:
    def test_backward_pass_runs(self):
        model = RouterClassifier(input_dim=Q_EMB_DIM + Q_FEAT_DIM + PROBE_DIM)
        x = torch.randn(4, Q_EMB_DIM + Q_FEAT_DIM + PROBE_DIM)
        y = torch.tensor([0, 1, 0, 1])
        loss = torch.nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
