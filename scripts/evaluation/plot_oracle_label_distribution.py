#!/usr/bin/env python3
"""Grouped bar chart: Dense vs Graph counts per split (Train, Dev, Test)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_EDIR = str(Path(__file__).resolve().parent)
if _EDIR not in sys.path:
    sys.path.insert(0, _EDIR)
from _figure_utils import apply_default_style, savefig_pdf

logger = logging.getLogger(__name__)

# (display name, filename under data-dir)
SPLIT_SPECS: tuple[tuple[str, str], ...] = (
    ("Train", "labeled_train.jsonl"),
    ("Dev", "labeled_dev.jsonl"),
    ("Test", "labeled_test.jsonl"),
)

LABEL_ORDER = ["Dense", "Graph"]


def _count_labels_in_file(path: Path) -> tuple[int, int]:
    """Return (n_dense, n_graph)."""
    n_dense = n_graph = 0
    if not path.is_file():
        logger.warning("Missing %s — counts will be zero.", path)
        return 0, 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            lab = row.get("gold_label")
            if lab == "Dense":
                n_dense += 1
            elif lab == "Graph":
                n_graph += 1
    return n_dense, n_graph


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Oracle gold-label counts by split (grouped bar chart)."
    )
    p.add_argument(
        "--data-dir",
        default="data/training",
        help="Directory containing labeled_train.jsonl, labeled_dev.jsonl, labeled_test.jsonl.",
    )
    p.add_argument(
        "--output", type=Path, default=Path("figures/oracle_label_distribution.pdf")
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_dir = Path(args.data_dir)
    plot_rows: list[dict] = []
    for split_name, fname in SPLIT_SPECS:
        n_d, n_g = _count_labels_in_file(data_dir / fname)
        plot_rows.append(
            {"split": split_name, "gold_label": "Dense", "count": n_d}
        )
        plot_rows.append(
            {"split": split_name, "gold_label": "Graph", "count": n_g}
        )

    if sum(r["count"] for r in plot_rows) == 0:
        logger.error("No gold_label rows found under %s.", data_dir)
        return 1

    df = pd.DataFrame(plot_rows)
    df["split"] = pd.Categorical(
        df["split"], categories=[s for s, _ in SPLIT_SPECS], ordered=True
    )
    df["gold_label"] = pd.Categorical(
        df["gold_label"], categories=LABEL_ORDER, ordered=True
    )

    apply_default_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.barplot(
        data=df,
        x="split",
        y="count",
        hue="gold_label",
        hue_order=LABEL_ORDER,
        ax=ax,
        palette={"Dense": "steelblue", "Graph": "darkorange"},
    )
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.set_title("Oracle label distribution by split")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Gold label", loc="upper right")
    fig.tight_layout()
    savefig_pdf(fig, Path(args.output))
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
