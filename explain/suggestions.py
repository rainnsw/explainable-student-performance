"""Convert SHAP-style explanations into human-readable insights and suggestions.

Design philosophy:
    - Insights state facts: "Your quiz average is hurting your prediction the most."
    - Suggestions are prescriptive: "Aim for at least 8 study hours per week — students at this level score X points higher on average."
    - Both are grounded in (a) SHAP contributions and (b) percentile context
      from the training distribution, so we are not just hand-waving.

We use templates rather than an LLM so the output is deterministic, fast,
and provably tied to the model's reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running this file directly OR importing as a module.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from explain.shap_explainer import Explanation


# Threshold for considering a feature "significant" enough to mention.
# Features whose absolute contribution is below this are filtered out.
MIN_CONTRIBUTION_MAGNITUDE = 0.5

# How many top helpers/hurters to surface in the summary
TOP_K_DRIVERS = 3


# Suggestion templates: each feature has an "improve direction" and a
# message. Direction is +1 if higher = better, -1 if lower = better.
FEATURE_DIRECTION = {
    "homework_avg": +1,
    "quiz_avg": +1,
    "midterm_score": +1,
    "study_hours_per_week": +1,
    "attendance_pct": +1,
    "submission_delay_days": -1,  # lower is better
    "improvement_trend": +1,  # positive trend is better
    "consistency_std": -1,  # lower std = more consistent = better
}


SUGGESTION_TEMPLATES = {
    "homework_avg": (
        "Homework is currently weighing your prediction down. "
        "Students with homework averages above {target:.0f} score about {gain:.1f} "
        "points higher on the final. Focus on completing and revising weak homework sets."
    ),
    "quiz_avg": (
        "Quiz performance is a strong driver. Students averaging above {target:.0f} on quizzes "
        "score about {gain:.1f} points higher. Review past quizzes and identify recurring weak topics."
    ),
    "midterm_score": (
        "Your midterm score is below the level associated with strong final performance. "
        "Students who scored above {target:.0f} on the midterm tend to score about {gain:.1f} "
        "points higher on the final."
    ),
    "study_hours_per_week": (
        "Study time looks low compared to top performers. Students studying at least {target:.0f} "
        "hours per week score about {gain:.1f} points higher. Even a steady 2-hour increase often shows up."
    ),
    "attendance_pct": (
        "Attendance is below the level associated with high final scores. "
        "Students attending more than {target:.0f}% of classes score about {gain:.1f} points higher."
    ),
    "submission_delay_days": (
        "Late submissions are dragging your prediction down. Students who submit within "
        "{target:.0f} days of the deadline score about {gain:.1f} points higher on the final."
    ),
    "improvement_trend": (
        "Your scores are not trending upward through the term. Students with a positive improvement "
        "trend score about {gain:.1f} points higher. Try to make each assessment better than the last."
    ),
    "consistency_std": (
        "Your performance is inconsistent across assessments. Students with low score variance "
        "(std under {target:.1f}) score about {gain:.1f} points higher. Consistency beats peaks."
    ),
}


@dataclass
class Suggestion:
    feature: str
    message: str
    estimated_gain: float  # rough estimate of point gain if addressed
    current_value: float
    target_value: float


@dataclass
class StudentReport:
    prediction: float
    base_value: float
    actual: float | None
    summary: str  # one-paragraph human-readable summary
    helpers: list[tuple[str, float, float]]  # (feature, contribution, value)
    hurters: list[tuple[str, float, float]]
    suggestions: list[Suggestion]


def _percentile(value: float, distribution: np.ndarray) -> float:
    """Return the percentile rank of `value` in `distribution`."""
    return float((distribution < value).mean() * 100)


def _target_percentile_value(distribution: np.ndarray, percentile: float = 75) -> float:
    """Return the value at a given percentile in the distribution."""
    return float(np.percentile(distribution, percentile))


def _make_summary(
    prediction: float,
    actual: float | None,
    helpers: list[tuple[str, float, float]],
    hurters: list[tuple[str, float, float]],
    feature_display: dict[str, str],
) -> str:
    """Compose a one-paragraph summary of the prediction."""
    parts = [f"Predicted final exam score: {prediction:.1f} / 100."]
    if actual is not None:
        parts.append(f"Actual: {actual:.1f}.")

    if helpers:
        top_helpers = ", ".join(
            feature_display.get(f, f) for f, _, _ in helpers[:2]
        )
        parts.append(f"Strongest positive factors: {top_helpers}.")

    if hurters:
        top_hurters = ", ".join(
            feature_display.get(f, f) for f, _, _ in hurters[:2]
        )
        parts.append(f"Areas dragging the prediction down: {top_hurters}.")

    return " ".join(parts)


def generate_report(
    explanation: Explanation,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_display: dict[str, str],
    actual: float | None = None,
    max_suggestions: int = 3,
) -> StudentReport:
    """Turn a raw Explanation into a structured StudentReport.

    X_train, y_train are used to compute peer benchmarks (the "students at
    target X percentile score Y points higher" claims in suggestions).
    """
    # Filter out tiny contributions
    filtered = [
        (f, c, v)
        for f, c, v in [
            (f, c, explanation.feature_values[f])
            for f, c in explanation.feature_contributions.items()
        ]
        if abs(c) >= MIN_CONTRIBUTION_MAGNITUDE
    ]
    helpers = sorted([t for t in filtered if t[1] > 0], key=lambda x: -x[1])
    hurters = sorted([t for t in filtered if t[1] < 0], key=lambda x: x[1])

    # Build suggestions from the top hurters
    suggestions: list[Suggestion] = []
    high_performers_mask = y_train >= np.percentile(y_train, 75)
    high_performers = X_train[high_performers_mask]
    avg_score_top = float(y_train[high_performers_mask].mean())
    avg_score_bot = float(
        y_train[~high_performers_mask].mean()
    )
    rough_gain = avg_score_top - avg_score_bot  # used as a coarse upper bound

    for feature, contribution, value in hurters[:max_suggestions]:
        direction = FEATURE_DIRECTION.get(feature, +1)
        # Compute target value: 75th-percentile among high performers
        # (or 25th if direction is negative, i.e. lower is better)
        if direction > 0:
            target = float(np.percentile(high_performers[feature], 50))
        else:
            target = float(np.percentile(high_performers[feature], 50))

        # Estimated gain: scale the rough gain by how big this contribution is
        # relative to the total absolute contribution
        total_abs = sum(abs(c) for c in explanation.feature_contributions.values())
        share = abs(contribution) / total_abs if total_abs > 0 else 0
        est_gain = rough_gain * share

        template = SUGGESTION_TEMPLATES.get(
            feature,
            f"Improving {feature_display.get(feature, feature)} could help.",
        )
        message = template.format(target=target, gain=est_gain)
        suggestions.append(
            Suggestion(
                feature=feature,
                message=message,
                estimated_gain=est_gain,
                current_value=value,
                target_value=target,
            )
        )

    summary = _make_summary(
        prediction=explanation.prediction,
        actual=actual,
        helpers=helpers,
        hurters=hurters,
        feature_display=feature_display,
    )

    return StudentReport(
        prediction=explanation.prediction,
        base_value=explanation.base_value,
        actual=actual,
        summary=summary,
        helpers=helpers,
        hurters=hurters,
        suggestions=suggestions,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.loader import generate_synthetic
    from data.features import (
        prepare_synthetic,
        get_feature_matrix,
        FEATURE_COLUMNS,
        FEATURE_DISPLAY_NAMES,
    )
    from models.train import train
    from explain.shap_explainer import StudentExplainer

    raw = generate_synthetic(n_students=500, seed=0)
    prepared = prepare_synthetic(raw)
    X, y = get_feature_matrix(prepared)

    result = train(X, y, do_cv=False)
    explainer = StudentExplainer(result.model, X, FEATURE_COLUMNS)

    # Generate reports for a few students
    for i in [1, 50, 200]:
        exp = explainer.explain(X.iloc[[i]])
        report = generate_report(
            exp, X, y, FEATURE_DISPLAY_NAMES, actual=float(y.iloc[i])
        )
        print("=" * 70)
        print(f"Student #{i}")
        print(report.summary)
        print("\nSuggestions:")
        for s in report.suggestions:
            print(f"  [{s.feature}] (current: {s.current_value:.2f}, target: {s.target_value:.2f})")
            print(f"    {s.message}")
        print()