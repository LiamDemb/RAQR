#!/usr/bin/env python3
"""Grouped permutation importance for the all-signals router (accuracy drop)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

_EDIR = str(Path(__file__).resolve().parent)
if _EDIR not in sys.path:
    sys.path.insert(0, _EDIR)
from _constants import RAQR_SIGNALS
from _figure_utils import apply_default_style, savefig_pdf
from _router_eval import load_classifier_bundle

from raqr.routers import SignalConfig
from raqr.routers.dataset import RouterDataset
from raqr.routers.signal_config import PROBE_KEYS, Q_EMB_DIM, Q_FEAT_KEYS

logger = logging.getLogger(__name__)


def _stack_xy(
    split_path: str,
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray, SignalConfig, nn.Module]:
    cfg = SignalConfig.from_str(RAQR_SIGNALS)
    model, scaler, stored = load_classifier_bundle(model_dir, cfg)
    ds = RouterDataset(split_path, stored, scaler=scaler)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append(x.numpy())
        ys.append(int(y.item()))
    X = np.stack(xs, axis=0)
    y = np.array(ys, dtype=np.int64)
    return X, y, stored, model


def _accuracy(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=device)
        pred = model(xt).argmax(dim=1).cpu().numpy()
    return float((pred == y).mean())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Permutation importance (accuracy drop) for scalar + Q-Emb block."
    )
    p.add_argument("--split-path", default="data/training/labeled_test.jsonl")
    p.add_argument("--model-dir", default="models")
    p.add_argument(
        "--output", type=Path, default=Path("figures/router_permutation_importance.pdf")
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-repeats", type=int, default=10)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rng = np.random.default_rng(args.seed)
    model_dir = Path(args.model_dir)

    X, y, stored, model = _stack_xy(args.split_path, model_dir)
    device = next(model.parameters()).device

    baseline = _accuracy(model, X, y, device)
    n_scalars = len(Q_FEAT_KEYS) + len(PROBE_KEYS)
    assert (
        X.shape[1] == Q_EMB_DIM + n_scalars
    ), f"Expected {Q_EMB_DIM + n_scalars} features, got {X.shape[1]}"

    names = list(Q_FEAT_KEYS) + list(PROBE_KEYS) + ["question_embedding (384-d)"]
    drops: list[float] = []

    for j in range(n_scalars):
        col = Q_EMB_DIM + j
        accs: list[float] = []
        for _ in range(args.n_repeats):
            Xp = X.copy()
            Xp[:, col] = rng.permutation(Xp[:, col])
            accs.append(_accuracy(model, Xp, y, device))
        drops.append(baseline - float(np.mean(accs)))

    accs_emb: list[float] = []
    for _ in range(args.n_repeats):
        Xp = X.copy()
        perm = rng.permutation(len(Xp))
        Xp[:, :Q_EMB_DIM] = Xp[perm, :Q_EMB_DIM]
        accs_emb.append(_accuracy(model, Xp, y, device))
    drops.append(baseline - float(np.mean(accs_emb)))

    apply_default_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    order = np.argsort(drops)[::-1]
    y_pos = np.arange(len(names))
    arr = np.array(drops)[order]
    lbls = [names[i] for i in order]
    ax.barh(y_pos, arr, color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(lbls)
    ax.invert_yaxis()
    ax.set_xlabel("Drop in accuracy (baseline − permuted)")
    ax.set_title(
        f"Permutation importance — {stored.identifier} (baseline acc={baseline:.3f})"
    )
    fig.tight_layout()
    savefig_pdf(fig, Path(args.output))
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
