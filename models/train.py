"""Train a gradient-boosted regressor on student performance features.

Primary backend: LightGBM (recommended).
Fallback backend: sklearn's HistGradientBoostingRegressor, used automatically
if LightGBM is not installed. The fallback exists so the codebase can be
tested in restricted environments; results will be very similar but the
LightGBM path is the canonical one for SHAP integration.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

# Try LightGBM, fall back to sklearn
try:
    import lightgbm as lgb

    BACKEND = "lightgbm"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor

    BACKEND = "sklearn_hgb"


@dataclass
class TrainResult:
    model: object
    backend: str
    feature_names: list[str]
    train_metrics: dict
    test_metrics: dict
    cv_metrics: dict


def _make_model():
    """Construct an untrained model using the available backend."""
    if BACKEND == "lightgbm":
        return lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
    else:
        return HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
        )


def _metrics(y_true, y_pred) -> dict:
    """Compute MAE, RMSE, R2."""
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def cross_validate(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
    """K-fold CV on MAE, RMSE, R2."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = {"mae": [], "rmse": [], "r2": []}
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        m = _make_model()
        m.fit(X_tr, y_tr)
        preds = m.predict(X_val)
        m_scores = _metrics(y_val.values, preds)
        for k, v in m_scores.items():
            scores[k].append(v)
    return {
        f"{k}_mean": float(np.mean(v)) for k, v in scores.items()
    } | {
        f"{k}_std": float(np.std(v)) for k, v in scores.items()
    }


def train(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    do_cv: bool = True,
) -> TrainResult:
    """Train a model on (X, y) and return metrics + the fitted estimator."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = _make_model()
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_metrics = _metrics(y_train.values, train_preds)
    test_metrics = _metrics(y_test.values, test_preds)
    cv_metrics = cross_validate(X, y) if do_cv else {}

    return TrainResult(
        model=model,
        backend=BACKEND,
        feature_names=list(X.columns),
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        cv_metrics=cv_metrics,
    )


def save(result: TrainResult, path: str | Path) -> None:
    """Persist a trained model + metadata to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(result, f)


def load(path: str | Path) -> TrainResult:
    with Path(path).open("rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.loader import generate_synthetic
    from data.features import prepare_synthetic, get_feature_matrix

    raw = generate_synthetic(n_students=1000, seed=0)
    prepared = prepare_synthetic(raw)
    X, y = get_feature_matrix(prepared)

    result = train(X, y)
    print(f"Backend: {result.backend}")
    print(f"Train metrics: {result.train_metrics}")
    print(f"Test  metrics: {result.test_metrics}")
    print(f"CV    metrics: {result.cv_metrics}")