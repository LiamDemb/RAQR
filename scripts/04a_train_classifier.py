#!/usr/bin/env python3
"""Train a router classifier (XGBoost) for a given signal configuration.

Usage:
    poetry run python scripts/04a_train_classifier.py --signals q_emb,probe
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
from sklearn.metrics import f1_score

from raqr.routers import RouterClassifier, RouterDataset, SignalConfig

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_PATH = "data/training/labeled_train.jsonl"
DEFAULT_DEV_PATH = "data/training/labeled_dev.jsonl"
DEFAULT_OUTPUT_DIR = "models"
DEFAULT_RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a router classifier (XGBoost).")
    p.add_argument(
        "--signals",
        required=True,
        help="Comma-separated signal groups: q_emb, q_feat, probe",
    )
    p.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    p.add_argument("--dev-path", default=DEFAULT_DEV_PATH)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--min-child-weight", type=float, default=1.0)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--reg-alpha", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=40,
        help="Stop if dev logloss does not improve for this many boosting rounds",
    )
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """Inverse-frequency class weights for imbalanced labels (one weight per class)."""
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.sum() * len(weights)
    return weights


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    np.random.seed(args.seed)

    config = SignalConfig.from_str(args.signals)
    logger.info("Signal config: %s (dim=%d)", config.identifier, config.input_dim)

    train_ds = RouterDataset(args.train_path, config)
    dev_ds = RouterDataset(args.dev_path, config, scaler=train_ds.scaler)

    X_train = train_ds.numpy_features
    y_train = train_ds.numpy_labels
    X_dev = dev_ds.numpy_features
    y_dev = dev_ds.numpy_labels

    class_w = compute_class_weights(y_train)
    sample_weight = class_w[y_train]

    model = RouterClassifier(
        config=config,
        random_state=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        gamma=args.gamma,
        n_jobs=args.n_jobs,
        # XGBoost 2.x: early stopping is configured on the estimator, not fit().
        early_stopping_rounds=args.early_stopping_rounds,
    )

    logger.info(
        "Training %s | %d train, %d dev | n_estimators=%d max_depth=%d lr=%.3f "
        "early_stopping_rounds=%d",
        config.identifier,
        len(train_ds),
        len(dev_ds),
        args.n_estimators,
        args.max_depth,
        args.learning_rate,
        args.early_stopping_rounds,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_dev, y_dev)],
        verbose=False,
    )

    dev_preds = model.predict(X_dev)
    best_dev_f1 = float(
        f1_score(y_dev, dev_preds, average="macro", zero_division=0)
    )
    logger.info(
        "Finished | best_iteration=%s | dev_macro_f1=%.4f",
        getattr(model.estimator, "best_iteration", None),
        best_dev_f1,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"classifier_{config.identifier}.pkl"
    payload = {
        "backend": "xgboost",
        "model": model.estimator,
        "config_identifier": config.identifier,
        "config_flags": {
            "use_q_emb": config.use_q_emb,
            "use_q_feat": config.use_q_feat,
            "use_probe": config.use_probe,
        },
        "xgb_params": model.xgb_params,
        "feature_dim": int(X_train.shape[1]),
        "best_dev_macro_f1": best_dev_f1,
    }
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
    logger.info("Saved model → %s (dev Macro-F1=%.4f)", model_path, best_dev_f1)

    scaler_path = out_dir / f"scaler_{config.identifier}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(train_ds.scaler, f)
    logger.info("Saved scaler → %s", scaler_path)


if __name__ == "__main__":
    main()
