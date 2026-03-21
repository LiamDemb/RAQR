#!/usr/bin/env python3
"""Validate trained router classifiers against gate metrics.

Gate metrics (must-pass):
  1. Dev accuracy > majority-class baseline
  2. Macro-F1 > 0 (non-collapsed predictions)
  3. Save confusion matrix to results/

Usage:
    poetry run python scripts/04b_validate_classifier.py --signals q_emb,probe
    poetry run python scripts/04b_validate_classifier.py --all
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from raqr.routers import (
    LABEL_NAMES,
    RouterClassifier,
    RouterDataset,
    SignalConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_DEV_PATH = "data/training/labeled_dev.jsonl"
DEFAULT_MODEL_DIR = "models"
DEFAULT_RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

ALL_ABLATIONS = [
    "q_emb",
    "q_feat",
    "probe",
    "q_emb,q_feat",
    "q_emb,probe",
    "q_feat,probe",
    "q_emb,q_feat,probe",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate router classifier gate metrics.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--signals",
        help="Comma-separated signal groups: q_emb, q_feat, probe",
    )
    group.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Validate all 7 ablation classifiers and produce summary",
    )
    p.add_argument("--dev-path", default=DEFAULT_DEV_PATH)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args(argv)


def majority_class_accuracy(labels: np.ndarray) -> float:
    counts = Counter(labels.tolist())
    return max(counts.values()) / len(labels) if len(labels) > 0 else 0.0


def save_confusion_matrix_plot(
    cm: np.ndarray, identifier: str, results_dir: Path
) -> Optional[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"Confusion Matrix — {identifier}")
        fig.colorbar(im, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(LABEL_NAMES)
        ax.set_yticklabels(LABEL_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        fig.tight_layout()
        fig_path = results_dir / f"confusion_{identifier}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        return fig_path
    except ImportError:
        return None


def validate_single(
    config: SignalConfig,
    dev_path: str,
    model_dir: Path,
    results_dir: Path,
    batch_size: int,
) -> dict:
    """Validate a single signal config. Returns a results dict."""
    model_path = model_dir / f"classifier_{config.identifier}.pt"
    scaler_path = model_dir / f"scaler_{config.identifier}.pkl"

    if not model_path.exists():
        logger.warning("Model not found: %s — skipping", model_path)
        return {
            "identifier": config.identifier,
            "status": "SKIPPED",
            "reason": "model not found",
        }

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    dev_ds = RouterDataset(dev_path, config, scaler=scaler)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)

    model = RouterClassifier(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        dropout=checkpoint.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for x, y in dev_loader:
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    dense_f1 = f1_score(y_true, y_pred, pos_label=0, average="binary", zero_division=0)
    graph_f1 = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
    baseline = majority_class_accuracy(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    report = classification_report(
        y_true, y_pred, target_names=LABEL_NAMES, zero_division=0
    )

    gate_acc = acc > baseline
    gate_f1 = macro_f1 > 0
    gate_pass = gate_acc and gate_f1

    # Save confusion matrix plot
    fig_path = save_confusion_matrix_plot(cm, config.identifier, results_dir)

    # Save per-config gate report
    status = "PASSED" if gate_pass else "FAILED"
    report_path = results_dir / f"gate_report_{config.identifier}.txt"
    with open(report_path, "w") as f:
        f.write(f"Config: {config.identifier}\n")
        f.write(f"Dev Accuracy: {acc:.4f}\n")
        f.write(f"Majority Baseline: {baseline:.4f}\n")
        f.write(f"Macro-F1: {macro_f1:.4f}\n")
        f.write(f"Gate: {status}\n\n")
        f.write(report)
        f.write(f"\nConfusion Matrix:\n{cm}\n")

    return {
        "identifier": config.identifier,
        "accuracy": acc,
        "baseline": baseline,
        "macro_f1": macro_f1,
        "dense_f1": dense_f1,
        "graph_f1": graph_f1,
        "gate_acc": "✓" if gate_acc else "✗",
        "gate_f1": "✓" if gate_f1 else "✗",
        "status": status,
        "confusion": cm,
        "report_path": str(report_path),
        "fig_path": str(fig_path) if fig_path else None,
    }


def print_summary_table(results: list[dict], results_dir: Path) -> None:
    """Print and save a consolidated ablation results table."""
    header = (
        f"{'Signals':<25} {'Acc':>6} {'Base':>6} {'M-F1':>6} "
        f"{'D-F1':>6} {'G-F1':>6} {'Gate':>8}"
    )
    sep = "─" * len(header)

    lines = [
        "",
        "═" * len(header),
        "  Phase 4 — Ablation Validation Summary",
        "═" * len(header),
        "",
        header,
        sep,
    ]

    for r in results:
        if r["status"] == "SKIPPED":
            lines.append(f"{r['identifier']:<25} {'— skipped (model not found) —'}")
            continue
        lines.append(
            f"{r['identifier']:<25} "
            f"{r['accuracy']:>5.3f} "
            f"{r['baseline']:>6.3f} "
            f"{r['macro_f1']:>5.3f} "
            f"{r['dense_f1']:>6.3f} "
            f"{r['graph_f1']:>5.3f} "
            f"  {r['status']:>6}"
        )

    passed = sum(1 for r in results if r["status"] == "PASSED")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    skipped = sum(1 for r in results if r["status"] == "SKIPPED")

    lines.append(sep)
    lines.append(f"  {passed} passed, {failed} failed, {skipped} skipped")
    lines.append("═" * len(header))
    lines.append("")

    output = "\n".join(lines)
    print(output)

    summary_path = results_dir / "ablation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(output)
    logger.info("Saved summary → %s", summary_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.run_all:
        signal_list = ALL_ABLATIONS
    else:
        signal_list = [args.signals]

    results = []
    for sig in signal_list:
        config = SignalConfig.from_str(sig)
        logger.info("Validating: %s", config.identifier)
        r = validate_single(config, args.dev_path, model_dir, results_dir, args.batch_size)
        results.append(r)

        if r["status"] != "SKIPPED":
            logger.info(
                "  %s | acc=%.3f baseline=%.3f macro-f1=%.3f → %s",
                r["identifier"], r["accuracy"], r["baseline"],
                r["macro_f1"], r["status"],
            )

    print_summary_table(results, results_dir)


if __name__ == "__main__":
    main()
