#!/usr/bin/env python3
"""Train a router classifier for a given signal configuration.

Usage:
    poetry run python scripts/04a_train_classifier.py --signals q_emb,probe --epochs 50
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from raqr.routers import RouterClassifier, RouterDataset, SignalConfig

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_PATH = "data/training/labeled_train.jsonl"
DEFAULT_DEV_PATH = "data/training/labeled_dev.jsonl"
DEFAULT_OUTPUT_DIR = "models"
DEFAULT_RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a router classifier.")
    p.add_argument(
        "--signals",
        required=True,
        help="Comma-separated signal groups: q_emb, q_feat, probe",
    )
    p.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    p.add_argument("--dev-path", default=DEFAULT_DEV_PATH)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def compute_class_weights(dataset: RouterDataset) -> torch.Tensor:
    """Inverse-frequency class weights for imbalanced labels."""
    labels = dataset._labels
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: RouterClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: RouterClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Returns (loss, accuracy, macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
        n += len(y)

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / max(n, 1)
    macro_f1 = float(
        f1_score(all_labels, all_preds, average="macro", zero_division=0)
    )
    return total_loss / max(n, 1), acc, macro_f1


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = SignalConfig.from_str(args.signals)
    logger.info("Signal config: %s (dim=%d)", config.identifier, config.input_dim)

    train_ds = RouterDataset(args.train_path, config)
    dev_ds = RouterDataset(args.dev_path, config, scaler=train_ds.scaler)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RouterClassifier(
        config=config,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout,
    ).to(device)

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    logger.info(
        "Training %s on %s | %d train, %d dev | AdamW lr=%.0e wd=%.0e dropout=%.1f",
        config.identifier, device, len(train_ds), len(dev_ds),
        args.lr, args.weight_decay, args.dropout,
    )

    best_dev_f1 = -1.0
    best_state: dict | None = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        dev_loss, dev_acc, dev_f1 = evaluate(model, dev_loader, criterion, device)

        improved = dev_f1 > best_dev_f1
        if improved:
            best_dev_f1 = dev_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch <= 5 or epoch % 10 == 0 or improved or patience_counter >= args.patience:
            logger.info(
                "Epoch %3d | train_loss=%.4f | dev_loss=%.4f | dev_acc=%.3f | "
                "dev_macro_f1=%.3f%s",
                epoch, train_loss, dev_loss, dev_acc, dev_f1,
                " *" if improved else "",
            )

        if patience_counter >= args.patience:
            logger.info(
                "Early stopping at epoch %d (best dev Macro-F1=%.4f)", epoch, best_dev_f1
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"classifier_{config.identifier}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config_identifier": config.identifier,
            # Full config flags stored so the validator can rebuild the same architecture
            "config_flags": {
                "use_q_emb": config.use_q_emb,
                "use_q_feat": config.use_q_feat,
                "use_probe": config.use_probe,
            },
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "best_dev_macro_f1": best_dev_f1,
        },
        model_path,
    )
    logger.info("Saved model → %s (best dev Macro-F1=%.4f)", model_path, best_dev_f1)

    scaler_path = out_dir / f"scaler_{config.identifier}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(train_ds.scaler, f)
    logger.info("Saved scaler → %s", scaler_path)


if __name__ == "__main__":
    main()
