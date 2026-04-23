"""
kovaak_tracker/dashboard.py
============================
Step 3: Interactive Visualizer
Displays the Tension Quadrant and Frame-by-Frame VOD review.
"""

import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Aim Performance Dashboard", layout="wide")

OUTPUT_DIR = Path("output")
metrics_path = OUTPUT_DIR / "metrics.json"
frames_path = OUTPUT_DIR / "frame_errors.csv"

if not metrics_path.exists():
    st.error("No analysis data found. Please run Analyze.py first.")
    st.stop()

with open(metrics_path) as f:
    metrics = json.load(f)

df = pd.read_csv(frames_path) if frames_path.exists() else None

st.title("🌊 Aim Tension & Speed Matching Analysis")
t = metrics['tension']

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Pure Tension Coeff (PTC)", f"{t['ptc']} Hz²")
    st.metric("Mean Relative Accel", f"{t['mean_a_rel']} px/s²")
    st.metric("Global Spatial Error", f"{t['avg_error_px']} px")

    l = metrics['loss']
    st.metric("Accuracy", f"{l['on_target_pct']} %")
    st.write(f"Loss: {l['loss_count']} times / {l['total_off_time']} s off-target")

with col2:
    fig = go.Figure()
    # Ideal line: a_rel proportional to error
    max_e = max(t['avg_error_px'] * 2, 1)
    fig.add_trace(go.Scatter(
        x=[0, max_e], y=[0, max_e * t['ptc']],
        mode='lines', name='Your Tension Slope',
        line=dict(dash='dash', color='gray')
    ))
    fig.add_trace(go.Scatter(
        x=[t['avg_error_px']], y=[t['mean_a_rel']],
        mode='markers', name='Your Result',
        marker=dict(size=15, color='cyan')
    ))
    fig.update_layout(
        title="Tension Diagnosis (PTC = a_rel / E)",
        xaxis_title="Spatial Error E (px)",
        yaxis_title="Relative Acceleration a_rel (px/s²)"
    )
    st.plotly_chart(fig, use_container_width=True)

if df is not None:
    st.subheader("Frame-by-Frame Error")
    st.line_chart(df.set_index("time_s")["error_px"])