from pathlib import Path

import streamlit as st

from kovaak_tracker.dashboard_data import build_error_timeline, load_dashboard_data, render_review_frame
from kovaak_tracker.video import save_uploaded_video


st.set_page_config(page_title="KovaaK Analyzer Dashboard", layout="wide")

metrics, df = load_dashboard_data()
if metrics is None or df is None:
    st.warning("Analysis data not found. Please run Analyze.py first.")
    st.stop()

st.title("KovaaK Tracking Dashboard")
st.caption("Mechanics diagnostics from tracking error, speed mismatch, and acceleration mismatch.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy (On-Target)", f"{metrics['loss']['on_target_pct']}%")
col2.metric("Global Avg Error", f"{metrics['tension']['avg_error_px']} px")
col3.metric("Loss Count", f"{metrics['loss']['loss_count']}")
col4.metric("Total Off-Target Time", f"{metrics['loss']['total_off_time']} s")

st.divider()

st.subheader("Kinematics Diagnostics")
tension = metrics["tension"]
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Speed Mismatch (dV)", f"{tension['speed_mismatch']} px/s", help="Magnitude of velocity difference.")
    st.caption("Lower means smoother tracking.")
with col2:
    st.metric("Accel Mismatch (dA)", f"{tension['accel_mismatch']} px/s^2", help="Magnitude of acceleration difference.")
    st.caption("Lower means more controlled tension.")
with col3:
    st.metric("Pure Tension Coeff (PTC)", f"{tension['ptc']} Hz^2", help="Acceleration mismatch divided by spatial error.")
    st.caption("Use this to compare sessions.")

st.divider()

st.subheader("Spatial Error Timeline")
st.plotly_chart(build_error_timeline(df), use_container_width=True)

st.divider()

st.subheader("Precision VOD Review")
video_file = st.file_uploader("Upload raw video (mp4/avi/mov/webm)", type=["mp4", "avi", "mov", "webm"])

if video_file:
    video_path = save_uploaded_video(video_file, Path(video_file.name).suffix)
    frame_idx = st.slider("Scrub to view tracked frames", 0, len(df) - 1, 0)
    frame = render_review_frame(video_path, df.iloc[frame_idx])
    if frame is not None:
        st.image(frame, use_container_width=True)
