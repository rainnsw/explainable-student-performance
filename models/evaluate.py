"""Model evaluation utilities.

Standalone module for computing metrics, running cross-validation, and
producing diagnostic plots / dataframes. The training module re-exports
some of these for convenience, but this is the canonical location.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


@dataclass
class EvalMetrics:
    mae: float
    rmse: float
    r2: float
    mape: float | None  # mean absolute percentage error (None if any y == 0)

    def as_dict(self) -> dict:
        return {"mae": self.mae, "rmse": self.rmse, "r2": self.r2, "mape": self.mape}


def compute_metrics(y_true, y_pred) -> EvalMetrics:
    """Compute MAE, RMSE, R2, and MAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # MAPE only if no zero values in y_true (else division blows up)
    if np.all(y_true != 0):
        mape = float(np.mean(np.abs(err / y_true)) * 100)
    else:
        mape = None

    return EvalMetrics(mae=mae, rmse=rmse, r2=r2, mape=mape)


def cross_validate(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """K-fold cross-validation.

    `model_factory` is a callable returning a fresh untrained estimator.
    Returns a dict with mean/std for each metric across folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = {"mae": [], "rmse": [], "r2": []}

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = model_factory()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        m = compute_metrics(y_val, preds)
        fold_scores["mae"].append(m.mae)
        fold_scores["rmse"].append(m.rmse)
        fold_scores["r2"].append(m.r2)

    return {
        f"{k}_mean": float(np.mean(v)) for k, v in fold_scores.items()
    } | {
        f"{k}_std": float(np.std(v)) for k, v in fold_scores.items()
    }


def residuals_dataframe(y_true, y_pred, X: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a tidy residuals dataframe useful for diagnostic plots.

    Columns: actual, predicted, residual, abs_residual, [feature columns].
    """
    df = pd.DataFrame(
        {
            "actual": np.asarray(y_true, dtype=float),
            "predicted": np.asarray(y_pred, dtype=float),
        }
    )
    df["residual"] = df["actual"] - df["predicted"]
    df["abs_residual"] = df["residual"].abs()
    if X is not None:
        df = pd.concat([df.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    return df


def error_buckets(y_true, y_pred, n_buckets: int = 4) -> pd.DataFrame:
    """Group predictions into buckets by predicted score and report metrics per bucket.

    Useful for spotting if the model is systematically worse at the extremes.
    """
    df = pd.DataFrame(
        {
            "actual": np.asarray(y_true, dtype=float),
            "predicted": np.asarray(y_pred, dtype=float),
        }
    )
    df["bucket"] = pd.qcut(df["predicted"], q=n_buckets, duplicates="drop")
    rows = []
    for bucket, group in df.groupby("bucket", observed=True):
        m = compute_metrics(group["actual"], group["predicted"])
        rows.append(
            {
                "bucket": str(bucket),
                "n": len(group),
                "mae": m.mae,
                "rmse": m.rmse,
                "r2": m.r2,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.loader import generate_synthetic
    from data.features import prepare_synthetic, get_feature_matrix
    from models.train import train

    df = prepare_synthetic(generate_synthetic(n_students=1000, seed=42))
    X, y = get_feature_matrix(df)
    result = train(X, y, do_cv=False)

    preds = result.model.predict(X)
    metrics = compute_metrics(y, preds)
    print(f"In-sample metrics: {metrics.as_dict()}")

    print("\nError by predicted-score bucket (lets you see if extremes are harder):")
    print(error_buckets(y, preds, n_buckets=4).round(2).to_string(index=False))