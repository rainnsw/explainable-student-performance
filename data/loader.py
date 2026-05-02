"""Data loading and synthesis for student performance prediction.

Provides:
- load_uci(): downloads and parses the UCI Student Performance dataset
- generate_synthetic(): creates a richer synthetic dataset designed to
  exercise the academic / behavioral / learning-pattern feature buckets
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"


# --------------------------------------------------------------------------
# UCI loader
# --------------------------------------------------------------------------

def load_uci(
    subject: Literal["math", "portuguese"] = "math",
    cache_dir: str | Path = "data/cache",
) -> pd.DataFrame:
    """Load the UCI Student Performance dataset.

    The dataset has G1, G2, G3 (three grade points). We treat G3 as the
    final exam score (target) and G1, G2 as the in-course assessments.
    """
    import urllib.request

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "student.zip"

    if not cache_file.exists():
        urllib.request.urlretrieve(UCI_URL, cache_file)

    inner = "student-mat.csv" if subject == "math" else "student-por.csv"
    with zipfile.ZipFile(cache_file) as zf:
        with zf.open(inner) as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), sep=";")

    return df


# --------------------------------------------------------------------------
# Synthetic generator
# --------------------------------------------------------------------------

def generate_synthetic(
    n_students: int = 1000,
    n_quizzes: int = 6,
    n_homeworks: int = 10,
    seed: int = 42,
    noise: float = 4.0,
) -> pd.DataFrame:
    """Generate a synthetic student performance dataset with rich features.

    The data-generating process:
        - Each student has latent ability (truncated normal, 0..100)
        - Each student has a latent diligence factor (0..1)
        - Behavioral signals (study hours, attendance, submission delay)
          are correlated with diligence
        - Quiz scores follow ability + diligence-based trend + noise
        - Homework average is driven by diligence + ability
        - Final exam score = nonlinear function of all of the above + noise

    The function returns a flat dataframe with the engineered features
    already computed (improvement_trend, consistency, etc.) so it can be
    used as a drop-in alongside the UCI data.
    """
    rng = np.random.default_rng(seed)

    # Latent factors
    ability = np.clip(rng.normal(60, 15, n_students), 5, 100)
    diligence = np.clip(rng.beta(2, 2, n_students), 0.05, 0.98)

    # Behavioral features (correlated with diligence)
    study_hours = np.clip(rng.normal(diligence * 12, 3), 0, 30)
    attendance = np.clip(rng.normal(60 + diligence * 35, 6), 30, 100)
    submission_delay = np.clip(
        rng.normal((1 - diligence) * 4, 1.2), 0, 14
    )  # days late on average

    # Quiz time series (n_quizzes per student)
    # Each student has a "trend" (improvement or decline) tied to diligence
    base_quiz = ability + (diligence - 0.5) * 10
    trend_per_quiz = (diligence - 0.4) * 1.5  # roughly -0.6 to +0.9 per quiz
    quiz_matrix = np.zeros((n_students, n_quizzes))
    for q in range(n_quizzes):
        quiz_matrix[:, q] = np.clip(
            base_quiz + trend_per_quiz * q + rng.normal(0, noise, n_students),
            0,
            100,
        )

    # Homework average (driven by diligence + a bit of ability)
    hw_avg = np.clip(
        diligence * 70 + ability * 0.3 + rng.normal(0, 5, n_students), 0, 100
    )

    # Midterm (correlated with quiz performance + ability)
    midterm = np.clip(
        quiz_matrix.mean(axis=1) * 0.6
        + ability * 0.4
        + rng.normal(0, noise, n_students),
        0,
        100,
    )

    # Engineered features
    quiz_avg = quiz_matrix.mean(axis=1)
    quiz_std = quiz_matrix.std(axis=1)  # consistency: low std = consistent
    # improvement trend: slope of linear fit across quiz scores
    quiz_indices = np.arange(n_quizzes)
    improvement_trend = np.array(
        [np.polyfit(quiz_indices, quiz_matrix[i], 1)[0] for i in range(n_students)]
    )

    # Final exam: nonlinear combination
    final_exam = (
        0.30 * midterm
        + 0.25 * quiz_avg
        + 0.15 * hw_avg
        + 0.10 * attendance
        + 0.08 * (study_hours / 30 * 100)
        - 0.05 * (submission_delay / 14 * 100)
        + 0.07 * (improvement_trend * 10)  # rewarded for upward trend
    )
    # Penalize inconsistency (high quiz_std) mildly
    final_exam = final_exam - 0.3 * quiz_std
    final_exam = np.clip(final_exam + rng.normal(0, noise * 0.7, n_students), 0, 100)

    df = pd.DataFrame(
        {
            # Academic
            "homework_avg": hw_avg.round(2),
            "quiz_avg": quiz_avg.round(2),
            "midterm_score": midterm.round(2),
            # Behavioral
            "study_hours_per_week": study_hours.round(2),
            "attendance_pct": attendance.round(2),
            "submission_delay_days": submission_delay.round(2),
            # Learning patterns
            "improvement_trend": improvement_trend.round(3),
            "consistency_std": quiz_std.round(3),
            # Target
            "final_exam_score": final_exam.round(2),
        }
    )

    # Also expose the raw quiz columns for downstream feature engineering
    for q in range(n_quizzes):
        df[f"quiz_{q + 1}"] = quiz_matrix[:, q].round(2)

    return df


if __name__ == "__main__":
    # Smoke test
    df = generate_synthetic(n_students=20, seed=0)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nFinal exam score: mean={df.final_exam_score.mean():.2f}, "
          f"std={df.final_exam_score.std():.2f}")