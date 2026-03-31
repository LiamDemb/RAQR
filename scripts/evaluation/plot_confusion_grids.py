#!/usr/bin/env python3
"""Side-by-side confusion matrices for router ablations (2x2 and 4x2 grid of 7)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_EDIR = str(Path(__file__).resolve().parent)
if _EDIR not in sys.path:
    sys.path.insert(0, _EDIR)
from _constants import ABLATION_GRID_FOUR, ALL_ABLATIONS
from _figure_utils import apply_default_style, savefig_pdf
from _router_eval import evaluate_split

from raqr.routers import LABEL_NAMES

logger = logging.getLogger(__name__)


def _count_nonempty_jsonl_lines(path: str) -> int:
    n = 0
    p = Path(path)
    if not p.is_file():
        return 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _plot_cm_grid(
    identifiers: list[str],
    split_path: str,
    model_dir: Path,
    batch_size: int,
    ncols: int,
    figsize: tuple[float, float],
) -> plt.Figure:
    """Plot confusion grids: one colorbar per panel, same numeric scale on all panels.

    Uses ``vmax = n_test`` so the scale is valid for imbalanced dev/test splits (train-only
    undersampling leaves test with natural class proportions).
    """
    n = len(identifiers)
    nrows = int(np.ceil(n / ncols))

    n_test = _count_nonempty_jsonl_lines(split_path)
    vmax_axis = float(n_test) if n_test else 1.0

    apply_default_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for idx, sig in enumerate(identifiers):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        try:
            ev = evaluate_split(sig, split_path, model_dir, batch_size=batch_size)
        except FileNotFoundError as e:
            logger.warning("%s", e)
            ax.set_visible(False)
            continue

        cm = ev["confusion"]
        im = ax.imshow(
            cm,
            interpolation="nearest",
            cmap=plt.cm.Blues,
            vmin=0.0,
            vmax=vmax_axis,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(ev["identifier"])
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(LABEL_NAMES)
        ax.set_yticklabels(LABEL_NAMES)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(int(cm[i, j])),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > vmax_axis / 2.0 else "black",
                )

    for j in range(len(identifiers), nrows * ncols):
        r, c = j // ncols, j % ncols
        axes[r][c].set_visible(False)

    fig.suptitle(
        f"Router confusion matrices (shared scale 0–{vmax_axis:g}; n_test={n_test})"
    )
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Confusion matrix grids for ablations.")
    p.add_argument("--split-path", default="data/training/labeled_test.jsonl")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--output-four", type=Path, default=Path("figures/confusion_matrix_grid_4.pdf")
    )
    p.add_argument(
        "--output-seven", type=Path, default=Path("figures/confusion_matrix_grid_7.pdf")
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    model_dir = Path(args.model_dir)

    fig4 = _plot_cm_grid(
        ABLATION_GRID_FOUR,
        args.split_path,
        model_dir,
        args.batch_size,
        ncols=2,
        figsize=(8, 8),
    )
    savefig_pdf(fig4, Path(args.output_four))

    fig7 = _plot_cm_grid(
        ALL_ABLATIONS,
        args.split_path,
        model_dir,
        args.batch_size,
        ncols=4,
        figsize=(14, 8),
    )
    savefig_pdf(fig7, Path(args.output_seven))
    logger.info("Wrote %s and %s", args.output_four, args.output_seven)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
