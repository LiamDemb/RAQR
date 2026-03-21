"""PyTorch dataset for router classifier training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .signal_config import (
    LABEL_MAP,
    PROBE_KEYS,
    Q_FEAT_KEYS,
    SignalConfig,
)

logger = logging.getLogger(__name__)


class RouterDataset(Dataset):
    """Loads labeled JSONL and assembles feature vectors per SignalConfig.

    Q-Emb vectors are already unit-normalized from the embedder.
    Q-Feat and Probe scalars are Z-score normalized via a fitted StandardScaler.
    """

    def __init__(
        self,
        path: str | Path,
        config: SignalConfig,
        scaler: Optional[StandardScaler] = None,
    ) -> None:
        self.config = config
        self.path = Path(path)

        rows = self._load_jsonl(self.path)
        if not rows:
            raise ValueError(f"No valid rows loaded from {self.path}")

        self._labels = np.array(
            [LABEL_MAP[r["gold_label"]] for r in rows], dtype=np.int64
        )

        q_emb = self._extract_q_emb(rows) if config.use_q_emb else None
        scalars = self._extract_scalars(rows)

        if scaler is not None:
            self._scaler = scaler
            if scalars is not None:
                scalars = self._scaler.transform(scalars)
        else:
            self._scaler = StandardScaler()
            if scalars is not None:
                scalars = self._scaler.fit_transform(scalars)

        parts = []
        if q_emb is not None:
            parts.append(q_emb)
        if scalars is not None:
            parts.append(scalars)

        self._features = np.hstack(parts).astype(np.float32)

        logger.info(
            "Loaded %d rows from %s | config=%s | feature_dim=%d",
            len(self._features),
            self.path.name,
            config.identifier,
            self._features.shape[1],
        )

    @property
    def scaler(self) -> StandardScaler:
        return self._scaler

    @property
    def numpy_features(self) -> np.ndarray:
        """Feature matrix for sklearn/XGBoost training (same order as `__getitem__`)."""
        return self._features

    @property
    def numpy_labels(self) -> np.ndarray:
        """Integer labels 0=Dense, 1=Graph."""
        return self._labels

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self._features[idx], dtype=torch.float32)
        y = torch.tensor(self._labels[idx], dtype=torch.long)
        return x, y

    def _load_jsonl(self, path: Path) -> list[dict]:
        rows = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON at line %d", i + 1)
                    continue

                if row.get("gold_label") not in LABEL_MAP:
                    logger.warning(
                        "Unknown label '%s' at line %d, skipping",
                        row.get("gold_label"),
                        i + 1,
                    )
                    continue

                self._validate_signals(row, i + 1)
                rows.append(row)
        return rows

    def _validate_signals(self, row: dict, line_num: int) -> None:
        """Fail fast if requested signal fields are missing."""
        if self.config.use_q_emb and "question_embedding" not in row:
            raise KeyError(
                f"Line {line_num}: 'question_embedding' missing but use_q_emb=True"
            )
        if self.config.use_q_feat:
            for key in Q_FEAT_KEYS:
                if key not in row:
                    raise KeyError(
                        f"Line {line_num}: '{key}' missing but use_q_feat=True"
                    )
        if self.config.use_probe:
            for key in PROBE_KEYS:
                if key not in row:
                    raise KeyError(
                        f"Line {line_num}: '{key}' missing but use_probe=True"
                    )

    def _extract_q_emb(self, rows: list[dict]) -> np.ndarray:
        return np.array(
            [r["question_embedding"] for r in rows], dtype=np.float32
        )

    def _extract_scalars(self, rows: list[dict]) -> Optional[np.ndarray]:
        """Extract Q-Feat and/or Probe scalar features for normalization."""
        keys: list[str] = []
        if self.config.use_q_feat:
            keys.extend(Q_FEAT_KEYS)
        if self.config.use_probe:
            keys.extend(PROBE_KEYS)
        if not keys:
            return None
        return np.array(
            [[r[k] for k in keys] for r in rows], dtype=np.float32
        )
