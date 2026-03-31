"""Load trained router checkpoints and run batched inference (mirrors 04b_validate_classifier)."""

from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from raqr.routers import RouterClassifier, RouterDataset, SignalConfig


def load_classifier_bundle(
    model_dir: Path,
    config: SignalConfig,
    device: torch.device | None = None,
) -> tuple[RouterClassifier, object, SignalConfig]:
    """Load scaler + RouterClassifier from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = model_dir / f"classifier_{config.identifier}.pt"
    scaler_path = model_dir / f"scaler_{config.identifier}.pkl"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    if "config_flags" in checkpoint:
        stored_config = SignalConfig(**checkpoint["config_flags"])
    else:
        stored_config = config

    model = RouterClassifier(
        config=stored_config,
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint.get("dropout", 0.5),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, scaler, stored_config


def predict_labels(
    model: RouterClassifier,
    dataset: RouterDataset,
    *,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) as int64 arrays."""
    if device is None:
        device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.tolist())
    return np.array(all_labels, dtype=np.int64), np.array(all_preds, dtype=np.int64)


def metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {"accuracy": acc, "macro_f1": macro_f1, "confusion": cm}


def evaluate_split(
    signals: str,
    split_path: str,
    model_dir: Path,
    batch_size: int = 64,
) -> dict:
    """Validate one SignalConfig on a labeled JSONL split; returns metrics + arrays."""
    cfg = SignalConfig.from_str(signals)
    model, scaler, stored_config = load_classifier_bundle(model_dir, cfg)
    ds = RouterDataset(split_path, stored_config, scaler=scaler)
    y_true, y_pred = predict_labels(
        model, ds, batch_size=batch_size, device=next(model.parameters()).device
    )
    m = metrics_from_arrays(y_true, y_pred)
    m["y_true"] = y_true
    m["y_pred"] = y_pred
    m["identifier"] = stored_config.identifier
    return m
