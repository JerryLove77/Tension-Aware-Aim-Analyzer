"""
kovaak_tracker/dashboard.py
============================
Data Dashboard and Online VOD Replay (Dual Mismatch Edition)
"""

import streamlit as st
import pandas as pd
import json
import cv2
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="KovaaK Analyzer Dashboard", layout="wide")
OUTPUT_DIR = Path("output")

@st.cache_data
def load_data():
    metrics_path = OUTPUT_DIR / "metrics.json"
    csv_path = OUTPUT_DIR / "frame_errors.csv"
    if not metrics_path.exists() or not csv_path.exists():
        return None, None
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    df = pd.read_csv(csv_path)
    if "chunk_id" not in df.columns: df["chunk_id"] = 0
    return metrics, df

metrics, df = load_data()
if metrics is None:
    st.warning("Analysis data not found. Please run Analyze.py first.")
    st.stop()

st.title("🎯 KovaaK Tracking Dashboard (Mechanics Lab)")
st.caption("Analyzing Muscle Tension and Tracking Smoothness via Vector Mismatch.")

# ── 1. Hit Performance ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy (On-Target)", f"{metrics['loss']['on_target_pct']}%")
col2.metric("Global Avg Error", f"{metrics['tension']['avg_error_px']} px")
col3.metric("Loss Count", f"{metrics['loss']['loss_count']}")
col4.metric("Total Off-Target Time", f"{metrics['loss']['total_off_time']} s")

st.divider()

# ── 2. Physics Metrics (The Dual Mismatch) ──
st.subheader("🌊 Kinematics Diagnostics (Loss Frames Only)")
t = metrics['tension']

col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    st.metric("Speed Mismatch (\u0394V)", f"{t['speed_mismatch']} px/s", help="Magnitude of velocity difference. Indicates tracking lag/overshoot.")
    st.caption("Lower = Smoother tracking.")
with col_t2:
    st.metric("Accel Mismatch (\u0394A)", f"{t['accel_mismatch']} px/s²", help="Magnitude of acceleration difference. Indicates violent micro-adjustments.")
    st.caption("Lower = Relaxed, controlled tension.")
with col_t3:
    st.metric("Pure Tension Coeff (PTC)", f"{t['ptc']} Hz²", help="ΔA / Spatial Error. Represents physical spring stiffness of your arm.")
    st.caption("Optimal elasticity zone: 60 - 180.")

st.divider()

# ── 3. Frame-by-Frame Timeline ──
st.subheader("📈 Spatial Error Timeline")
fig = go.Figure()
colors = px.colors.qualitative.Plotly

for chunk_id, group in df.groupby("chunk_id"):
    color = colors[int(chunk_id) % len(colors)]
    fig.add_trace(go.Scatter(
        x=group["time_s"], y=group["error_px"], 
        mode='lines', name=f'Target {int(chunk_id) + 1}',
        line=dict(color=color, width=2)
    ))

off_target_df = df[df["on_target"] == 0]
if not off_target_df.empty:
    fig.add_trace(go.Scatter(
        x=off_target_df["time_s"], y=off_target_df["error_px"],
        mode='markers', name='Target Lost (Analysis Zone)',
        marker=dict(color='#EF233C', size=5, symbol='x')
    ))

fig.update_layout(xaxis_title="Time (s)", yaxis_title="Spatial Error (px)", 
                  hovermode="x unified", height=400, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── 4. Precision VOD Review ──
st.subheader("🎬 Precision VOD Review")
video_file = st.file_uploader("Upload raw video (mp4/avi/webm)", type=["mp4", "avi", "webm"])

if video_file:
    import tempfile
    file_extension = Path(video_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name
        
    cap = cv2.VideoCapture(video_path)
    frame_idx = st.slider("Scrub to view tracked frames", 0, len(df)-1, 0)
    row = df.iloc[frame_idx]
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        vis = frame.copy()
        cross_pos = (int(row["cross_x"]), int(row["cross_y"]))
        cv2.drawMarker(vis, cross_pos, (0, 180, 255), cv2.MARKER_CROSS, 20, 2)
        
        if pd.notna(row["ball_w"]):
            cx, cy = int(row["ball_x"]), int(row["ball_y"])
            w, h = int(row["ball_w"]), int(row["ball_h"])
            top_left = (int(cx - w / 2), int(cy - h / 2))
            bottom_right = (int(cx + w / 2), int(cy + h / 2))
            color = (0, 255, 0) if row["on_target"] == 1 else (0, 0, 255)
            
            cv2.rectangle(vis, top_left, bottom_right, color, 2)
            cv2.circle(vis, (cx, cy), 2, color, -1)
            cv2.line(vis, (cx, cy), cross_pos, color, 1)
        
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)