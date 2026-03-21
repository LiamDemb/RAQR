"""MLP classifier for routing queries to retrieval strategies."""

from __future__ import annotations

import torch
import torch.nn as nn


class RouterClassifier(nn.Module):
    """Lightweight MLP classifier with dynamic input dimension.

    Architecture:
        Linear(input_dim, hidden_dim) → ReLU → Dropout
        Linear(hidden_dim, hidden_dim) → ReLU → Dropout
        Linear(hidden_dim, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
