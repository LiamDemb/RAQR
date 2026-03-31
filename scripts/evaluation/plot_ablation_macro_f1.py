#!/usr/bin/env python3
"""Horizontal bar chart: macro-F1 for each of 7 router input ablations on a split."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_EDIR = str(Path(__file__).resolve().parent)
if _EDIR not in sys.path:
    sys.path.insert(0, _EDIR)
from _constants import ALL_ABLATIONS
from _figure_utils import apply_default_style, savefig_pdf
from _router_eval import evaluate_split

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ablation macro-F1 bar chart.")
    p.add_argument("--split-path", default="data/training/labeled_test.jsonl")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("figures/ablation_macro_f1.pdf"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    model_dir = Path(args.model_dir)
    rows: list[dict] = []

    for sig in ALL_ABLATIONS:
        try:
            r = evaluate_split(
                sig, args.split_path, model_dir, batch_size=args.batch_size
            )
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", sig, e)
            continue
        rows.append(
            {
                "signals": r["identifier"].replace("-", " + "),
                "macro_f1": r["macro_f1"],
            }
        )

    if not rows:
        logger.error("No models evaluated; nothing to plot.")
        return 1

    df = pd.DataFrame(rows)
    apply_default_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df = df.sort_values("macro_f1", ascending=True)
    sns.barplot(data=df, y="signals", x="macro_f1", ax=ax, color="steelblue")
    ax.set_xlabel("Macro-F1")
    ax.set_ylabel("Input configuration")
    ax.set_title("Signal ablation — router macro-F1")
    ax.set_xlim(0.65, 0.8)
    fig.tight_layout()
    savefig_pdf(fig, Path(args.output))
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
