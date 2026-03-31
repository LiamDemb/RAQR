#!/usr/bin/env python3
"""Bar chart: oracle gold label counts before vs after undersampling (seaborn)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_EDIR = str(Path(__file__).resolve().parent)
if _EDIR not in sys.path:
    sys.path.insert(0, _EDIR)
from _figure_utils import apply_default_style, savefig_pdf

logger = logging.getLogger(__name__)

SPLITS = ("labeled_train.jsonl", "labeled_dev.jsonl", "labeled_test.jsonl")


def _count_labels(router_dir: Path) -> Counter[str]:
    c: Counter[str] = Counter()
    for name in SPLITS:
        fp = router_dir / name
        if not fp.is_file():
            logger.warning("Missing %s — skipping.", fp)
            continue
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                lab = row.get("gold_label")
                if lab in ("Dense", "Graph"):
                    c[lab] += 1
    return c


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Oracle label distribution before/after undersampling."
    )
    p.add_argument(
        "--before-dir",
        required=True,
        help="Router dataset dir without undersampling (e.g. data/training_unbalanced).",
    )
    p.add_argument(
        "--after-dir",
        required=True,
        help="Router dataset dir with undersampling (e.g. data/training).",
    )
    p.add_argument(
        "--output", type=Path, default=Path("figures/oracle_label_distribution.pdf")
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    before_dir = Path(args.before_dir)
    after_dir = Path(args.after_dir)
    cb = _count_labels(before_dir)
    ca = _count_labels(after_dir)
    if not cb and not ca:
        logger.error("No labeled files found in either directory.")
        return 1

    before_rows: list[dict] = []
    for lab, n in cb.items():
        before_rows.extend(
            [{"stage": "Before undersampling", "gold_label": lab} for _ in range(n)]
        )
    after_rows: list[dict] = []
    for lab, n in ca.items():
        after_rows.extend(
            [{"stage": "After undersampling", "gold_label": lab} for _ in range(n)]
        )
    long_df = pd.DataFrame(before_rows + after_rows)
    if long_df.empty:
        logger.error("No gold_label rows to plot.")
        return 1
    apply_default_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(
        data=long_df,
        x="gold_label",
        hue="stage",
        order=["Dense", "Graph"],
        ax=ax,
    )
    ax.set_ylabel("Count")
    ax.set_xlabel("Gold label")
    ax.set_title("Oracle label distribution")
    fig.tight_layout()
    savefig_pdf(fig, Path(args.output))
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
