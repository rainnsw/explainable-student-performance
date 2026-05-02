"""SHAP-based explanations for student performance predictions.

Primary backend: SHAP's TreeExplainer, which gives exact Shapley values
for tree-based models (LightGBM, XGBoost, sklearn ensembles).

Fallback backend: when SHAP is unavailable, we approximate per-feature
contributions using a leave-one-feature-out (LOFO) style perturbation
against the feature's mean. This is *not* equivalent to SHAP — it ignores
feature interactions — but it preserves the API so the suggestion layer
and dashboard can be tested. In your final environment, SHAP will be
installed and the real backend will be used automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@dataclass
class Explanation:
    """Per-prediction explanation."""

    prediction: float
    base_value: float  # the model's expected output (mean prediction)
    feature_contributions: dict[str, float]  # feature -> SHAP value
    feature_values: dict[str, float]  # feature -> input value

    def top_drivers(self, k: int = 3) -> list[tuple[str, float, float]]:
        """Return top-k features by absolute contribution.

        Returns list of (feature_name, contribution, feature_value).
        """
        items = sorted(
            self.feature_contributions.items(),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:k]
        return [(name, contrib, self.feature_values[name]) for name, contrib in items]

    def helpers(self) -> list[tuple[str, float, float]]:
        """Features pushing the prediction UP, sorted by magnitude."""
        items = [
            (n, c, self.feature_values[n])
            for n, c in self.feature_contributions.items()
            if c > 0
        ]
        return sorted(items, key=lambda x: x[1], reverse=True)

    def hurters(self) -> list[tuple[str, float, float]]:
        """Features pushing the prediction DOWN, sorted by magnitude."""
        items = [
            (n, c, self.feature_values[n])
            for n, c in self.feature_contributions.items()
            if c < 0
        ]
        return sorted(items, key=lambda x: x[1])  # most negative first


class StudentExplainer:
    """Wraps a trained model and produces SHAP-based explanations."""

    def __init__(self, model, X_background: pd.DataFrame, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self.X_background = X_background

        if SHAP_AVAILABLE:
            # TreeExplainer works for LightGBM, XGBoost, sklearn ensembles
            self._explainer = shap.TreeExplainer(model)
            self._backend = "shap"
        else:
            # Fallback: precompute baseline prediction and per-feature means
            self._feature_means = X_background.mean()
            self._baseline = float(model.predict(X_background).mean())
            self._explainer = None
            self._backend = "fallback_lofo"

    @property
    def backend(self) -> str:
        return self._backend

    def explain(self, x: pd.DataFrame | pd.Series) -> Explanation:
        """Explain a single prediction.

        x can be a single-row DataFrame or a Series. Always returns one
        Explanation object.
        """
        if isinstance(x, pd.Series):
            x_df = x.to_frame().T[self.feature_names]
        else:
            x_df = x[self.feature_names].iloc[[0]] if len(x) > 0 else x

        prediction = float(self.model.predict(x_df)[0])

        if self._backend == "shap":
            shap_values = self._explainer.shap_values(x_df)
            # shap_values shape: (1, n_features) for regression
            contribs = dict(zip(self.feature_names, shap_values[0].tolist()))
            base = float(self._explainer.expected_value)
            if isinstance(self._explainer.expected_value, np.ndarray):
                base = float(self._explainer.expected_value[0])
        else:
            # Fallback: contribution = prediction(x) - prediction(x with feature replaced by mean)
            # This is a single-feature perturbation, NOT true Shapley
            contribs = {}
            for feat in self.feature_names:
                x_perturbed = x_df.copy()
                x_perturbed[feat] = self._feature_means[feat]
                pred_without = float(self.model.predict(x_perturbed)[0])
                contribs[feat] = prediction - pred_without
            base = self._baseline

        feature_values = {f: float(x_df[f].iloc[0]) for f in self.feature_names}

        return Explanation(
            prediction=prediction,
            base_value=base,
            feature_contributions=contribs,
            feature_values=feature_values,
        )

    def explain_batch(self, X: pd.DataFrame) -> list[Explanation]:
        """Explain multiple predictions. Slower but useful for dashboards."""
        return [self.explain(X.iloc[[i]]) for i in range(len(X))]


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.loader import generate_synthetic
    from data.features import prepare_synthetic, get_feature_matrix, FEATURE_COLUMNS
    from models.train import train

    raw = generate_synthetic(n_students=500, seed=0)
    prepared = prepare_synthetic(raw)
    X, y = get_feature_matrix(prepared)

    result = train(X, y, do_cv=False)
    explainer = StudentExplainer(result.model, X, FEATURE_COLUMNS)
    print(f"Explainer backend: {explainer.backend}")

    # Explain a few predictions
    for i in [0, 1, 50, 100]:
        exp = explainer.explain(X.iloc[[i]])
        print(f"\nStudent {i}: actual={y.iloc[i]:.2f}, predicted={exp.prediction:.2f}")
        print(f"  Base value: {exp.base_value:.2f}")
        print("  Top 3 drivers:")
        for name, contrib, val in exp.top_drivers(3):
            sign = "+" if contrib >= 0 else ""
            print(f"    {name}: value={val:.2f}, contribution={sign}{contrib:.2f}")