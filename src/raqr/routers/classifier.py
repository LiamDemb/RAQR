"""MLP classifier for routing queries to retrieval strategies."""

from __future__ import annotations

import torch
import torch.nn as nn
import os

from raqr.routers.signal_config import Q_EMB_DIM, SignalConfig

EMB_COMPRESSED_DIM = int(
    os.getenv("EMB_COMPRESSED_DIM", 32)
)  # Bottleneck output size for Q-Emb


class RouterClassifier(nn.Module):
    """Late-fusion MLP classifier with an optional Q-Emb bottleneck.

    When Q-Emb is active, the 384-dim embedding is first compressed to a
    `EMB_COMPRESSED_DIM`-dimensional representation via a linear bottleneck
    before being concatenated with scalar signals (Q-Feat / Probe). This
    prevents the high-dimensional embeddings from drowning out the scalar
    features in the MLP head.

    Architecture (when all signals active):
        Q-Emb (384) → Linear(384, 16) → LeakyReLU   ┐
        Scalars (7)  → [pass-through]               ┴→ cat(23) → MLP head → 2
    """

    def __init__(
        self,
        config: SignalConfig,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.config = config

        # Bottleneck: only exists when Q-Emb is an active signal
        self.emb_compressor: nn.Linear | None = None
        if config.use_q_emb:
            self.emb_compressor = nn.Linear(Q_EMB_DIM, EMB_COMPRESSED_DIM)

        # Head input dim = compressed emb (or 0) + all scalar features
        emb_out = EMB_COMPRESSED_DIM if config.use_q_emb else 0
        head_input_dim = emb_out + config.num_scalar_features

        # LeakyReLU preserves negative semantic directions from embeddings
        self.bottleneck_act = nn.LeakyReLU(negative_slope=0.1)
        self.net = nn.Sequential(
            nn.Linear(head_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []

        offset = 0
        if self.config.use_q_emb:
            emb = x[:, offset : offset + Q_EMB_DIM]
            parts.append(self.bottleneck_act(self.emb_compressor(emb)))  # type: ignore[arg-type]
            offset += Q_EMB_DIM

        # Remaining dimensions are scalar features (Q-Feat and/or Probe)
        if offset < x.shape[1]:
            parts.append(x[:, offset:])

        combined = torch.cat(parts, dim=1)
        return self.net(combined)
