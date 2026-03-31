#!/usr/bin/env python3
"""Grouped bar chart: mean token-F1 on test for Always Dense/Graph, RAQR, Perfect Oracle."""

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
from _constants import RAQR_SIGNALS
from _figure_utils import apply_default_style, savefig_pdf
from _router_eval import evaluate_split

from raqr.evaluation.metrics import compute_max_f1

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
        description="End-to-end system token-F1 (four routing policies)."
    )
    p.add_argument("--labeled-test", default="data/training/labeled_test.jsonl")
    p.add_argument("--model-dir", default="models")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("figures/e2e_system_f1.pdf"))
    p.add_argument("--output-json", type=Path, default=None)
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
            RAQR_SIGNALS,
            args.labeled_test,
            model_dir,
            batch_size=args.batch_size,
        )
    except FileNotFoundError as e:
        logger.error("RAQR model not found: %s", e)
        return 1

    y_pred = ev["y_pred"]
    if len(y_pred) != len(rows):
        logger.error(
            "Prediction count mismatch: %d vs %d rows.", len(y_pred), len(rows)
        )
        return 1

    f1_always_dense: list[float] = []
    f1_always_graph: list[float] = []
    f1_raqr: list[float] = []
    f1_oracle: list[float] = []

    for i, row in enumerate(rows):
        golds = _normalize_golds(row.get("gold_answers"))
        pd_ = str(row.get("pred_dense") or "")
        pg_ = str(row.get("pred_graph") or "")
        gl = row.get("gold_label")
        f1_always_dense.append(compute_max_f1(pd_, golds))
        f1_always_graph.append(compute_max_f1(pg_, golds))
        pred_route = int(y_pred[i])
        chosen = pd_ if pred_route == 0 else pg_
        f1_raqr.append(compute_max_f1(chosen, golds))
        if gl == "Graph":
            f1_oracle.append(compute_max_f1(pg_, golds))
        else:
            f1_oracle.append(compute_max_f1(pd_, golds))

    means = {
        "Always Dense": float(sum(f1_always_dense) / len(f1_always_dense)),
        "Always Graph": float(sum(f1_always_graph) / len(f1_always_graph)),
        "RAQR System": float(sum(f1_raqr) / len(f1_raqr)),
        "Perfect Oracle": float(sum(f1_oracle) / len(f1_oracle)),
    }

    if args.output_json:
        out = {"n": len(rows), "mean_token_f1": means}
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as jf:
            json.dump(out, jf, indent=2)
        logger.info("Wrote %s", args.output_json)

    df = pd.DataFrame([{"policy": k, "mean_token_f1": v} for k, v in means.items()])
    order = ["Always Dense", "Always Graph", "RAQR System", "Perfect Oracle"]
    df["policy"] = pd.Categorical(df["policy"], categories=order, ordered=True)
    df = df.sort_values("policy")

    apply_default_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="policy", y="mean_token_f1", ax=ax, color="steelblue")
    ax.set_ylabel("Mean token F1")
    ax.set_xlabel("")
    ax.set_title("End-to-end system performance (test set)")
    ax.set_ylim(0.2, 0.8)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    savefig_pdf(fig, Path(args.output))
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
