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
# [Omitted data loading logic]

st.title("🌊 Aim Tension & Speed Matching Analysis")
t = metrics['tension']

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Tension Score", f"{t['tension_score']} / 100")
    st.info(f"Diagnosis: {t['diagnosis']}")
    st.write(f"TBR Ratio: {t['tension_ratio']}x")

with col2:
    # Quadrant Chart: Visualizing the J/E relationship
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 3], y=[0, 3], mode='lines', name='Ideal Matching', line=dict(dash='dash', color='gray')))
    fig.add_trace(go.Scatter(x=[t['avg_error_px']/20], y=[t['jitter_rate']/15000], 
                             mode='markers', marker=dict(size=15, color='cyan')))
    fig.update_layout(title="Tension Diagnosis Quadrant", xaxis_title="Normalized Error (E)", yaxis_title="Normalized Jitter (J)")
    st.plotly_chart(fig, use_container_width=True)