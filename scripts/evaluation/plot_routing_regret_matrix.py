#!/usr/bin/env python3
"""2x2 heatmap: mean token-F1 lost per (true label, predicted label) for the RAQR router."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_EDIR = str(Path(__file__).resolve().parent)
if _EDIR not in sys.path:
    sys.path.insert(0, _EDIR)
from _constants import RAQR_SIGNALS
from _figure_utils import apply_default_style, savefig_pdf
from _router_eval import evaluate_split

from raqr.evaluation.metrics import compute_max_f1
from raqr.routers import LABEL_NAMES

logger = logging.getLogger(__name__)


def _normalize_golds(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [str(raw).strip()] if str(raw).strip() else []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Routing regret severity: mean F1 lost per confusion cell (seaborn heatmap)."
    )
    p.add_argument("--labeled-test", default="data/training/labeled_test.jsonl")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--signals", default=RAQR_SIGNALS)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--output", type=Path, default=Path("figures/routing_regret_severity.pdf")
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    model_dir = Path(args.model_dir)

    rows: list[dict] = []
    with open(args.labeled_test, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        logger.error("No rows in %s", args.labeled_test)
        return 1

    try:
        ev = evaluate_split(
            args.signals,
            args.labeled_test,
            model_dir,
            batch_size=args.batch_size,
        )
    except FileNotFoundError as e:
        logger.error("Router model not found: %s", e)
        return 1

    y_true = ev["y_true"]
    y_pred = ev["y_pred"]
    if len(y_true) != len(rows) or len(y_pred) != len(rows):
        logger.error("Length mismatch between JSONL and predictions.")
        return 1

    sum_regret = np.zeros((2, 2), dtype=np.float64)
    counts = np.zeros((2, 2), dtype=np.int64)

    for i, row in enumerate(rows):
        golds = _normalize_golds(row.get("gold_answers"))
        pd_ = str(row.get("pred_dense") or "")
        pg_ = str(row.get("pred_graph") or "")
        gl = row.get("gold_label")
        f1_oracle = (
            compute_max_f1(pg_, golds) if gl == "Graph" else compute_max_f1(pd_, golds)
        )
        route = int(y_pred[i])
        chosen = pd_ if route == 0 else pg_
        f1_chosen = compute_max_f1(chosen, golds)
        regret = max(0.0, f1_oracle - f1_chosen)
        t, p = int(y_true[i]), int(y_pred[i])
        sum_regret[t, p] += regret
        counts[t, p] += 1

    mean_regret = np.full((2, 2), np.nan, dtype=np.float64)
    for t in range(2):
        for p in range(2):
            if counts[t, p] > 0:
                mean_regret[t, p] = sum_regret[t, p] / counts[t, p]

    df = pd.DataFrame(mean_regret, index=LABEL_NAMES, columns=LABEL_NAMES)
    mask = np.isnan(mean_regret)

    m = np.nanmax(mean_regret)
    vmax_plot = float(m) if np.isfinite(m) and m > 0 else 1.0

    apply_default_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        df,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax_plot,
        mask=mask,
        cbar_kws={"label": "Mean token F1 lost"},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Routing regret severity (test set)")
    fig.tight_layout()
    savefig_pdf(fig, Path(args.output))
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
