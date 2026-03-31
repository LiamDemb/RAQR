#!/usr/bin/env python3
"""Permutation importance (accuracy drop) per signal block; all-combined + top ablation if different."""

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
from _constants import ALL_ABLATIONS, RAQR_SIGNALS
from _figure_utils import apply_default_style, savefig_pdf
from _router_eval import evaluate_split, load_classifier_bundle

from raqr.routers import SignalConfig
from raqr.routers.dataset import RouterDataset
from raqr.routers.signal_config import PROBE_KEYS, Q_EMB_DIM, Q_FEAT_KEYS

logger = logging.getLogger(__name__)


def _feature_blocks(cfg: SignalConfig) -> list[tuple[str, slice, bool]]:
    """(display name, column slice in X, is_row_shuffle_block)."""
    blocks: list[tuple[str, slice, bool]] = []
    col = 0
    if cfg.use_q_emb:
        blocks.append(
            ("question_embedding (384-d)", slice(col, col + Q_EMB_DIM), True)
        )
        col += Q_EMB_DIM
    if cfg.use_q_feat:
        for k in Q_FEAT_KEYS:
            blocks.append((k, slice(col, col + 1), False))
            col += 1
    if cfg.use_probe:
        for k in PROBE_KEYS:
            blocks.append((k, slice(col, col + 1), False))
            col += 1
    return blocks


def _stack_xy(
    signals: str,
    split_path: str,
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray, SignalConfig, nn.Module]:
    cfg = SignalConfig.from_str(signals)
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
    expected = stored.input_dim
    if X.shape[1] != expected:
        raise ValueError(
            f"Feature dim mismatch for {stored.identifier}: expected {expected}, got {X.shape[1]}"
        )
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


def _permute_block(
    Xp: np.ndarray,
    sl: slice,
    *,
    row_shuffle: bool,
    rng: np.random.Generator,
) -> None:
    if row_shuffle:
        perm = rng.permutation(len(Xp))
        Xp[:, sl] = Xp[perm, sl]
    else:
        c = sl.start
        Xp[:, c] = rng.permutation(Xp[:, c])


def compute_permutation_drops(
    signals: str,
    split_path: str,
    model_dir: Path,
    rng: np.random.Generator,
    n_repeats: int,
) -> tuple[float, list[str], list[float], str]:
    """Return baseline accuracy, feature names (same order as drops), drops, config identifier."""
    X, y, stored, model = _stack_xy(signals, split_path, model_dir)
    device = next(model.parameters()).device
    blocks = _feature_blocks(stored)

    baseline = _accuracy(model, X, y, device)
    names = [b[0] for b in blocks]
    drops: list[float] = []

    for _name, sl, row_shuffle in blocks:
        accs: list[float] = []
        for _ in range(n_repeats):
            Xp = X.copy()
            _permute_block(Xp, sl, row_shuffle=row_shuffle, rng=rng)
            accs.append(_accuracy(model, Xp, y, device))
        drops.append(baseline - float(np.mean(accs)))

    return baseline, names, drops, stored.identifier


def _plot_permutation_figure(
    baseline: float,
    names: list[str],
    drops: list[float],
    identifier: str,
) -> plt.Figure:
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
        f"Permutation importance — {identifier} (baseline acc={baseline:.3f})"
    )
    fig.tight_layout()
    return fig


def _same_signal_config(a: str, b: str) -> bool:
    return SignalConfig.from_str(a).identifier == SignalConfig.from_str(b).identifier


def _find_top_ablation_macro_f1(
    split_path: str,
    model_dir: Path,
    batch_size: int,
) -> tuple[str | None, float]:
    best_sig: str | None = None
    best_f1 = -1.0
    for sig in ALL_ABLATIONS:
        try:
            r = evaluate_split(sig, split_path, model_dir, batch_size=batch_size)
        except FileNotFoundError:
            logger.warning("Skipping ablation %s (model missing).", sig)
            continue
        f1 = float(r["macro_f1"])
        if f1 > best_f1:
            best_f1 = f1
            best_sig = sig
    return best_sig, best_f1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Permutation importance for all-signals router; "
        "also top macro-F1 ablation when it differs."
    )
    p.add_argument("--split-path", default="data/training/labeled_test.jsonl")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--output", type=Path, default=Path("figures/router_permutation_importance.pdf")
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-repeats", type=int, default=10)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rng = np.random.default_rng(args.seed)
    model_dir = Path(args.model_dir)
    out_path = Path(args.output)

    top_sig, top_f1 = _find_top_ablation_macro_f1(
        args.split_path, model_dir, args.batch_size
    )
    if top_sig is not None:
        logger.info(
            "Top macro-F1 ablation on split: %s (macro_f1=%.4f)",
            SignalConfig.from_str(top_sig).identifier,
            top_f1,
        )

    # 1) Always export all-combined (RAQR full router)
    try:
        base_c, names_c, drops_c, id_c = compute_permutation_drops(
            RAQR_SIGNALS,
            args.split_path,
            model_dir,
            rng,
            args.n_repeats,
        )
    except FileNotFoundError as e:
        logger.error("All-combined model required: %s", e)
        return 1

    fig_c = _plot_permutation_figure(base_c, names_c, drops_c, id_c)
    savefig_pdf(fig_c, out_path)
    logger.info("Wrote %s", out_path)

    # 2) If top ablation ≠ all-combined, export second chart
    if top_sig is not None and not _same_signal_config(top_sig, RAQR_SIGNALS):
        try:
            base_t, names_t, drops_t, id_t = compute_permutation_drops(
                top_sig,
                args.split_path,
                model_dir,
                rng,
                args.n_repeats,
            )
        except FileNotFoundError as e:
            logger.warning("Top ablation model missing for permutation plot: %s", e)
            return 0

        stem, suf = out_path.stem, out_path.suffix
        top_out = out_path.parent / f"{stem}_{id_t}{suf}"
        fig_t = _plot_permutation_figure(base_t, names_t, drops_t, id_t)
        savefig_pdf(fig_t, top_out)
        logger.info("Wrote %s (top macro-F1 config)", top_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
