"""Feature engineering for student performance data.

The goal of this module is to take either the UCI dataset or the synthetic
dataset and produce a unified feature matrix with the same column names,
so downstream code (training, SHAP, dashboard) does not need to branch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The canonical feature set used by the model.
FEATURE_COLUMNS = [
    # Academic
    "homework_avg",
    "quiz_avg",
    "midterm_score",
    # Behavioral
    "study_hours_per_week",
    "attendance_pct",
    "submission_delay_days",
    # Learning patterns
    "improvement_trend",
    "consistency_std",
]

TARGET_COLUMN = "final_exam_score"

# Human-readable names for the dashboard / suggestions
FEATURE_DISPLAY_NAMES = {
    "homework_avg": "Homework Average",
    "quiz_avg": "Quiz Average",
    "midterm_score": "Midterm Score",
    "study_hours_per_week": "Study Hours per Week",
    "attendance_pct": "Attendance %",
    "submission_delay_days": "Submission Delay (days)",
    "improvement_trend": "Improvement Trend",
    "consistency_std": "Consistency (lower = more consistent)",
}


def prepare_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic data already has the canonical features. Just select them."""
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()


def prepare_uci(df: pd.DataFrame) -> pd.DataFrame:
    """Map the UCI dataset onto the canonical feature schema.

    UCI dataset columns we use:
        G1, G2, G3       -> three sequential grade points (0-20 scale)
        studytime        -> 1=<2h, 2=2-5h, 3=5-10h, 4=>10h per week
        absences         -> number of absences (we convert to attendance %)
        failures         -> number of past class failures
        Walc / Dalc      -> alcohol consumption (proxy for diligence?)

    UCI has no homework/midterm/submission-delay columns, so we synthesize
    proxies from what is available. This is honest: we document the mapping
    rather than pretending UCI has things it does not.
    """
    out = pd.DataFrame()

    # Convert grades from 0-20 to 0-100 scale to match synthetic data
    g1 = df["G1"] * 5
    g2 = df["G2"] * 5
    g3 = df["G3"] * 5

    # Academic features
    # Homework avg: best proxy is G1 (the earliest assessment)
    out["homework_avg"] = g1
    # Quiz avg: use G2 as a midpoint assessment proxy
    out["quiz_avg"] = g2
    # Midterm: average of G1 and G2 (representing in-course performance)
    out["midterm_score"] = (g1 + g2) / 2

    # Behavioral features
    # studytime is ordinal 1-4; map to approximate hours per week
    studytime_map = {1: 1.0, 2: 3.5, 3: 7.5, 4: 12.0}
    out["study_hours_per_week"] = df["studytime"].map(studytime_map).fillna(3.5)

    # Attendance %: convert absences to attendance percentage
    # Assume ~80 class sessions per term as a rough denominator
    SESSIONS = 80
    out["attendance_pct"] = np.clip(
        100 * (1 - df["absences"].clip(0, SESSIONS) / SESSIONS), 0, 100
    )

    # Submission delay: UCI has no direct equivalent. Use `failures` (past
    # course failures) as a weak proxy, scaled into a plausible range.
    # 0 failures -> 0 days late, 3+ failures -> ~5 days late
    out["submission_delay_days"] = df["failures"].clip(0, 4) * 1.5

    # Learning patterns
    # Improvement trend: slope from G1 -> G2 (only two points, so simple diff)
    out["improvement_trend"] = (g2 - g1) / 1.0  # per "period"

    # Consistency: std dev of [G1, G2]
    out["consistency_std"] = pd.concat([g1, g2], axis=1).std(axis=1)

    # Target
    out[TARGET_COLUMN] = g3

    return out


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a prepared dataframe into X, y."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


if __name__ == "__main__":
    # Smoke test with synthetic data
    from loader import generate_synthetic

    raw = generate_synthetic(n_students=30, seed=0)
    prepared = prepare_synthetic(raw)
    X, y = get_feature_matrix(prepared)
    print("Synthetic prepared:")
    print(X.head())
    print(f"\nX shape: {X.shape}, y shape: {y.shape}")
    print(f"y stats: mean={y.mean():.2f}, std={y.std():.2f}")