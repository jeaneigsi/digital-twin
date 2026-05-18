from __future__ import annotations

import logging
from typing import override

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import SimulationConfig, default_config
from src.features import extract_from_events
from src.regime_detection import FEATURE_COLUMNS, RegimeDetector
from src.scheduler import ProductionScheduler, adapt_jobs_for_regime, generate_jobs
from src.simulation import ProductionLine, build_regime_schedule

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Production Digital Twin", layout="wide")
st.title("Production Regime Transition Digital Twin")


# ---------------------------------------------------------------------------
# Sidebar – Controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Simulation Parameters")

    nominal_dur = st.slider("Nominal duration", 100, 1000, 500, 50)
    degraded_dur = st.slider("Degraded duration", 100, 1000, 500, 50)
    recovery_dur = st.slider("Recovery duration", 100, 1000, 500, 50)
    window_size = st.slider("Feature window size", 20, 200, 50, 10)
    random_seed = st.number_input("Random seed", 0, 9999, 42)

    if st.button("Run Simulation", type="primary"):
        st.session_state.run_triggered = True
    else:
        st.session_state.setdefault("run_triggered", False)


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

if not st.session_state.run_triggered:
    st.info("Adjust parameters in the sidebar and click **Run Simulation**.")
    st.stop()

cfg = SimulationConfig(
    regimes=default_config().regimes,
    inter_arrival_time_mean=default_config().inter_arrival_time_mean,
    random_seed=random_seed,
    window_size=window_size,
)

# 1 — Simulation
with st.spinner("Running discrete-event simulation …"):
    line = ProductionLine(cfg)
    schedule = build_regime_schedule(nominal_dur, degraded_dur, recovery_dur)
    total_time = nominal_dur + degraded_dur + recovery_dur
    events = line.run(duration=total_time, regime_schedule=schedule)

# 2 — Features
features = extract_from_events(events, window_size)

# 3 — Regime detection
detector = RegimeDetector(random_state=random_seed)
detector.fit(features)
predictions = detector.predict(features)
probas = detector.predict_proba(features)

features["predicted"] = predictions
features["pred_label"] = ["nominal" if p == 0 else "degraded" for p in predictions]
features["confidence"] = probas.max(axis=1)

change_points = detector.detect_change_points(features)

# 4 — Scheduling
n_jobs = min(15, max(3, events["job_id"].nunique() // 5))
jobs = generate_jobs(n_jobs, cfg.machines_count)
scheduler = ProductionScheduler()

results = {}
for regime_label in ("nominal", "degraded"):
    regime_int = 0 if regime_label == "nominal" else 1
    adapted = adapt_jobs_for_regime(list(jobs), regime_int)
    results[regime_label] = scheduler.schedule(adapted, cfg.machines_count)

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

st.metric("Total Jobs", events["job_id"].nunique())
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Events", len(events))
col2.metric("Feature Windows", len(features))
col3.metric("Change Points", len(change_points))
col4.metric("Classifier Accuracy", f"{detector.classifier.score(detector.scaler.transform(features[FEATURE_COLUMNS].values.astype(np.float64)), features['predicted'].values):.2%}")

st.divider()

# -- Row 1: Time-series of indicators
st.subheader("Operational Indicators Over Time")

plot_df = events.copy()
plot_df["time_window"] = (plot_df["time"] // window_size * window_size).astype(int)

base = alt.Chart(plot_df).mark_line().encode(x=alt.X("time_window:Q", title="Time"))

chart_queue = base.encode(y="mean(queue_length):Q", color=alt.Color("regime:N")).properties(title="Avg Queue Length", height=200)
chart_proc = base.encode(y="mean(processing_time):Q", color=alt.Color("regime:N")).properties(title="Avg Processing Time", height=200)
chart_defect = base.encode(y="mean(is_defect):Q", color=alt.Color("regime:N")).properties(title="Defect Rate", height=200)

row1 = alt.hconcat(chart_queue, chart_proc, chart_defect).resolve_scale(color="shared")
st.altair_chart(row1, use_container_width=True)

# -- Row 2: Regime detection results
st.subheader("Regime Detection")

col_left, col_right = st.columns(2)

with col_left:
    detection_df = features[["time", "regime", "pred_label", "confidence"]].copy()
    chart_detection = alt.Chart(detection_df).mark_circle(size=60).encode(
        x="time:Q",
        y="pred_label:N",
        color=alt.Color("regime:N", title="True regime"),
        tooltip=["time", "regime", "pred_label", "confidence"],
    ).properties(title="Predicted vs True Regime", height=250)
    st.altair_chart(chart_detection, use_container_width=True)

with col_right:
    st.write("Change-point indices:", change_points if change_points else "None detected")
    accuracy = (features["regime"] == features["pred_label"]).mean()
    st.metric("Window-level Accuracy", f"{accuracy:.2%}")
    st.dataframe(features[["time", "regime", "pred_label", "confidence"]].head(20), use_container_width=True)

st.divider()

# -- Row 3: Scheduling comparison
st.subheader("Scheduling Adaptation")

col_nom, col_deg = st.columns(2)

with col_nom:
    st.markdown("**Nominal schedule**")
    if results["nominal"].status != "infeasible":
        schedule_nom = results["nominal"].schedule
        res = results["nominal"]
        chart_nom = alt.Chart(schedule_nom).mark_bar().encode(
            x="start:Q",
            x2="end:Q",
            y="machine:N",
            color="job_id:N",
            tooltip=["job_id", "machine", "start", "end"],
        ).properties(title=f"Makespan: {res.makespan} | Flow: {res.total_flow_time:.0f} ({res.status})", height=250)
        st.altair_chart(chart_nom, use_container_width=True)
    else:
        st.warning("Infeasible")

with col_deg:
    st.markdown("**Degraded-adapted schedule**")
    if results["degraded"].status != "infeasible":
        schedule_deg = results["degraded"].schedule
        res = results["degraded"]
        chart_deg = alt.Chart(schedule_deg).mark_bar().encode(
            x="start:Q",
            x2="end:Q",
            y="machine:N",
            color="job_id:N",
            tooltip=["job_id", "machine", "start", "end"],
        ).properties(title=f"Makespan: {res.makespan} | Flow: {res.total_flow_time:.0f} ({res.status})", height=250)
        st.altair_chart(chart_deg, use_container_width=True)
    else:
        st.warning("Infeasible")

# -- Row 4: Raw data
st.divider()
with st.expander("Raw simulation events (first 500 rows)"):
    st.dataframe(events.head(500), use_container_width=True)
