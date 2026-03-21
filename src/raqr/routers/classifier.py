"""XGBoost classifier for routing queries to retrieval strategies."""

from __future__ import annotations

from typing import Any

import numpy as np
from xgboost import XGBClassifier

from raqr.routers.signal_config import SignalConfig

_DEFAULT_XGB_KWARGS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "n_jobs": -1,
}


class RouterClassifier:
    """Gradient-boosted trees on concatenated Q-Emb / Q-Feat / Probe features."""

    def __init__(self, config: SignalConfig, **xgb_kwargs: Any) -> None:
        self.config = config
        merged = {**_DEFAULT_XGB_KWARGS, **xgb_kwargs}
        self._clf = XGBClassifier(**merged)
        self._xgb_params = merged

    @property
    def xgb_params(self) -> dict[str, Any]:
        return dict(self._xgb_params)

    @property
    def estimator(self) -> XGBClassifier:
        """Underlying sklearn-compatible XGBoost model (for checkpoint I/O)."""
        return self._clf

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        eval_set: list[tuple[np.ndarray, np.ndarray]] | None = None,
        verbose: bool | int = False,
    ) -> RouterClassifier:
        """Fit the booster. Early stopping uses ``early_stopping_rounds`` from init (XGBoost 2.x)."""
        kw: dict[str, Any] = {"verbose": verbose}
        if sample_weight is not None:
            kw["sample_weight"] = sample_weight
        if eval_set is not None:
            kw["eval_set"] = eval_set
        self._clf.fit(X_train, y_train, **kw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X)
