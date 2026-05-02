"""Streamlit dashboard for the Student Performance Prediction & Explanation System.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make project modules importable when running via `streamlit run`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

# --------------------------------------------------------------------------
# Defensive imports: surface errors clearly inside the page rather than
# crashing silently before any UI renders.
# --------------------------------------------------------------------------
try:
    from data.loader import generate_synthetic, load_uci
    from data.features import (
        prepare_synthetic,
        prepare_uci,
        get_feature_matrix,
        FEATURE_COLUMNS,
        FEATURE_DISPLAY_NAMES,
    )
    from models.train import train
    from explain.shap_explainer import StudentExplainer
    from explain.suggestions import generate_report
except Exception:
    st.error("❌ Failed to import project modules.")
    st.code(traceback.format_exc())
    st.stop()


# --------------------------------------------------------------------------
# Cached data + model loading
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading dataset…")
def load_dataset(source: str) -> pd.DataFrame:
    if source == "Synthetic (1000 students)":
        return prepare_synthetic(generate_synthetic(n_students=1000, seed=42))
    elif source == "UCI Math":
        return prepare_uci(load_uci("math"))
    elif source == "UCI Portuguese":
        return prepare_uci(load_uci("portuguese"))
    raise ValueError(f"Unknown source: {source}")


@st.cache_resource(show_spinner="Training model… (10–20 seconds on first run)")
def train_model_cached(dataset_name: str):
    """Train a model. Returns just the TrainResult — explainer is built fresh
    in the page so we don't try to cache complex objects together."""
    df = load_dataset(dataset_name)
    X, y = get_feature_matrix(df)
    return train(X, y, do_cv=True)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("🎓 Student Performance")
st.sidebar.markdown("Predicts final exam scores and explains the reasoning.")

dataset_name = st.sidebar.selectbox(
    "Dataset",
    ["Synthetic (1000 students)", "UCI Math", "UCI Portuguese"],
)

# Load data + model with explicit error handling
try:
    df = load_dataset(dataset_name)
    X, y = get_feature_matrix(df)
    result = train_model_cached(dataset_name)
    explainer = StudentExplainer(result.model, X, FEATURE_COLUMNS)
except Exception:
    st.error("❌ Failed to load data or train model.")
    st.code(traceback.format_exc())
    st.stop()

st.sidebar.markdown(f"**Model backend:** `{result.backend}`")
st.sidebar.markdown(f"**Explainer backend:** `{explainer.backend}`")
st.sidebar.markdown(f"**Dataset size:** {len(X)} students")
st.sidebar.success("✅ App loaded")


# --------------------------------------------------------------------------
# Main tabs
# --------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs(["🔍 Predict & Explain", "👥 Cohort View", "📊 Model Performance"])


# ----- Tab 1: Predict & Explain -----
with tab1:
    st.header("Predict & Explain")
    st.markdown(
        "Enter a student's profile or pick one from the dataset. "
        "The model predicts the final exam score and explains the main drivers."
    )

    mode = st.radio(
        "Input mode",
        ["Pick from dataset", "Enter custom values"],
        horizontal=True,
    )

    if mode == "Pick from dataset":
        idx = st.slider("Student index", 0, len(X) - 1, 0)
        x_input = X.iloc[[idx]]
        actual = float(y.iloc[idx])
    else:
        st.markdown("**Adjust the sliders to describe a student:**")
        col1, col2 = st.columns(2)
        custom = {}
        half = len(FEATURE_COLUMNS) // 2 + 1
        for col, feats in [(col1, FEATURE_COLUMNS[:half]), (col2, FEATURE_COLUMNS[half:])]:
            with col:
                for feat in feats:
                    fmin = float(X[feat].min())
                    fmax = float(X[feat].max())
                    fmean = float(X[feat].mean())
                    custom[feat] = st.slider(
                        FEATURE_DISPLAY_NAMES[feat],
                        min_value=fmin,
                        max_value=fmax,
                        value=fmean,
                        step=(fmax - fmin) / 100,
                    )
        x_input = pd.DataFrame([custom])[FEATURE_COLUMNS]
        actual = None

    explanation = explainer.explain(x_input)
    report = generate_report(explanation, X, y, FEATURE_DISPLAY_NAMES, actual=actual)

    # Headline metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Final Score", f"{report.prediction:.1f} / 100")
    m2.metric("Cohort Average", f"{float(y.mean()):.1f} / 100")
    if actual is not None:
        m3.metric(
            "Actual",
            f"{actual:.1f} / 100",
            delta=f"{report.prediction - actual:+.1f} (error)",
        )
    else:
        m3.metric("Model Base Value", f"{report.base_value:.1f}")

    st.markdown(f"**Summary:** {report.summary}")

    # Drivers chart — Plotly so it works on any Streamlit version
    st.subheader("What's driving this prediction?")
    contribs_df = pd.DataFrame(
        [
            {
                "Feature": FEATURE_DISPLAY_NAMES[f],
                "Contribution": c,
                "Value": explanation.feature_values[f],
            }
            for f, c in explanation.feature_contributions.items()
        ]
    ).sort_values("Contribution")

    chart_col, table_col = st.columns([2, 1])
    with chart_col:
        import plotly.graph_objects as go

        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in contribs_df["Contribution"]]
        fig = go.Figure(
            go.Bar(
                x=contribs_df["Contribution"],
                y=contribs_df["Feature"],
                orientation="h",
                marker_color=colors,
            )
        )
        fig.update_layout(
            xaxis_title="Contribution to prediction",
            yaxis_title="",
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    with table_col:
        st.dataframe(
            contribs_df[["Feature", "Value", "Contribution"]].round(2),
            hide_index=True,
            use_container_width=True,
        )

    # Suggestions
    st.subheader("📝 Suggestions for improvement")
    if not report.suggestions:
        st.success(
            "No major weak spots — this student's profile is already aligned with "
            "high final exam performance."
        )
    else:
        for i, s in enumerate(report.suggestions, 1):
            with st.expander(
                f"{i}. {FEATURE_DISPLAY_NAMES[s.feature]} "
                f"(current: {s.current_value:.1f}, target: {s.target_value:.1f})",
                expanded=(i == 1),
            ):
                st.write(s.message)
                st.caption(
                    f"Estimated impact if improved to target: ~{s.estimated_gain:+.1f} points"
                )


# ----- Tab 2: Cohort View -----
with tab2:
    st.header("Cohort View")
    st.markdown("Predictions across the full dataset. Filter and sort to find groups of interest.")

    preds = result.model.predict(X)
    cohort = X.copy()
    cohort["actual"] = y.values
    cohort["predicted"] = preds.round(2)
    cohort["residual"] = (cohort["actual"] - cohort["predicted"]).round(2)

    f1, f2 = st.columns(2)
    with f1:
        score_range = st.slider(
            "Predicted score range",
            float(cohort["predicted"].min()),
            float(cohort["predicted"].max()),
            (float(cohort["predicted"].min()), float(cohort["predicted"].max())),
        )
    with f2:
        sort_by = st.selectbox(
            "Sort by",
            [
                "predicted (high to low)",
                "predicted (low to high)",
                "residual (largest errors first)",
            ],
        )

    filtered = cohort[
        (cohort["predicted"] >= score_range[0]) & (cohort["predicted"] <= score_range[1])
    ]
    if sort_by == "predicted (high to low)":
        filtered = filtered.sort_values("predicted", ascending=False)
    elif sort_by == "predicted (low to high)":
        filtered = filtered.sort_values("predicted", ascending=True)
    else:
        filtered = filtered.sort_values("residual", key=lambda s: s.abs(), ascending=False)

    st.dataframe(filtered.round(2), use_container_width=True, height=400)

    st.subheader("Predicted vs. actual")
    st.scatter_chart(filtered, x="actual", y="predicted")


# ----- Tab 3: Model Performance -----
with tab3:
    st.header("Model Performance")

    st.subheader("Test-set metrics")
    cols = st.columns(3)
    cols[0].metric("MAE", f"{result.test_metrics['mae']:.2f}")
    cols[1].metric("RMSE", f"{result.test_metrics['rmse']:.2f}")
    cols[2].metric("R²", f"{result.test_metrics['r2']:.3f}")

    if result.cv_metrics:
        st.subheader("5-fold cross-validation")
        cv_df = pd.DataFrame(
            {
                "Metric": ["MAE", "RMSE", "R²"],
                "Mean": [
                    result.cv_metrics["mae_mean"],
                    result.cv_metrics["rmse_mean"],
                    result.cv_metrics["r2_mean"],
                ],
                "Std": [
                    result.cv_metrics["mae_std"],
                    result.cv_metrics["rmse_std"],
                    result.cv_metrics["r2_std"],
                ],
            }
        )
        st.dataframe(cv_df.round(3), hide_index=True, use_container_width=True)

    # Error by score bucket — diagnostic from evaluate.py
    st.subheader("Error by predicted-score bucket")
    st.caption("Reveals whether the model is systematically worse at the extremes.")
    try:
        from models.evaluate import error_buckets

        buckets = error_buckets(y, result.model.predict(X), n_buckets=4)
        st.dataframe(buckets.round(2), hide_index=True, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not compute error buckets: {e}")

    st.subheader("Global feature importance")
    st.markdown(
        "Mean absolute contribution across a sample of the dataset. "
        "Higher = the feature has more influence on predictions overall."
    )
    sample_size = min(200, len(X))
    sample = X.sample(sample_size, random_state=0)
    explanations = explainer.explain_batch(sample)
    importance = {feat: 0.0 for feat in FEATURE_COLUMNS}
    for exp in explanations:
        for feat, contrib in exp.feature_contributions.items():
            importance[feat] += abs(contrib)

    imp_df = pd.DataFrame(
        {
            "Feature": [FEATURE_DISPLAY_NAMES[f] for f in importance],
            "Mean |Contribution|": [v / sample_size for v in importance.values()],
        }
    ).sort_values("Mean |Contribution|", ascending=True)

    import plotly.graph_objects as go

    fig = go.Figure(
        go.Bar(
            x=imp_df["Mean |Contribution|"],
            y=imp_df["Feature"],
            orientation="h",
            marker_color="#3498db",
        )
    )
    fig.update_layout(
        xaxis_title="Mean |Contribution|",
        yaxis_title="",
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)