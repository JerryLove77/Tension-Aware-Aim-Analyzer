from pathlib import Path

import cv2
import streamlit as st

from kovaak_tracker.settings import ensure_output_dir
from kovaak_tracker.tracking import run_tracking_analysis
from kovaak_tracker.video import get_video_metadata, read_frame, save_uploaded_video
from kovaak_tracker.vision import frame_to_rgb, get_hsv_range, sample_median_bgr


st.set_page_config(page_title="KovaaK Tracker Calibration", layout="wide")
st.title("KovaaK Tracking Analyzer")
st.caption("Tracking-by-detection pipeline with OpenCV tracker fallback.")

ensure_output_dir()


def show_video_upload():
    video_file = st.file_uploader("Support mp4 / avi / mov / webm", type=["mp4", "avi", "mov", "webm"])
    if video_file is None:
        st.stop()

    suffix = Path(video_file.name).suffix
    video_path = save_uploaded_video(video_file, suffix)
    metadata = get_video_metadata(video_path)
    st.success(f"Video Loaded: {metadata.width}x{metadata.height} @ {metadata.fps:.1f}fps")
    return video_path, metadata


def select_video_range(frame_count: int) -> tuple[int, int]:
    st.header("1. Trim Valid Video Clip")
    return st.slider("Select Valid Video Range", 0, frame_count, (0, frame_count))


def select_target_color(video_path: str, start_frame: int, end_frame: int, width: int, height: int):
    st.header("2. Extract Target Color")
    frame_idx = st.slider("Slide to find the target ball", start_frame, max(start_frame, end_frame - 1), start_frame)
    selected_frame = read_frame(video_path, frame_idx)

    st.image(frame_to_rgb(selected_frame), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    x1 = col1.number_input("Left x1", 0, width, width // 2 - 10)
    y1 = col2.number_input("Top y1", 0, height, height // 2 - 10)
    x2 = col3.number_input("Right x2", 0, width, width // 2 + 10)
    y2 = col4.number_input("Bottom y2", 0, height, height // 2 + 10)

    if x2 <= x1 or y2 <= y1:
        st.stop()

    ball_bgr = sample_median_bgr(selected_frame, int(x1), int(y1), int(x2), int(y2))
    hsv_lo, hsv_hi = get_hsv_range(ball_bgr)

    preview = selected_frame.copy()
    cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    st.image(frame_to_rgb(preview), caption=f"Extracted RGB: {ball_bgr[::-1]}")

    return ball_bgr, hsv_lo, hsv_hi


video_path, metadata = show_video_upload()
start_frame, end_frame = select_video_range(metadata.frame_count)
if end_frame <= start_frame:
    st.warning("Please select a non-empty video range.")
    st.stop()

ball_bgr, hsv_lo, hsv_hi = select_target_color(
    video_path,
    start_frame,
    end_frame,
    metadata.width,
    metadata.height,
)

st.header("3. Start Analysis")

if st.button("Start Tracking Analysis", type="primary"):
    progress = st.progress(0, text="Initializing tracking engine...")

    def update_progress(value: float, text: str | None = None) -> None:
        progress.progress(value, text=text)

    tracking_run = run_tracking_analysis(
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        ball_bgr=ball_bgr,
        ball_hsv_lo=hsv_lo,
        ball_hsv_hi=hsv_hi,
        progress_callback=update_progress,
        warn_callback=st.warning,
    )
    progress.empty()

    stats = tracking_run.stats
    st.subheader("Tracking Engine Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("High-Speed Tracked Frames", stats["frames_tracked"], help="Frames handled by OpenCV tracker")
    col2.metric("Color Detected Frames", stats["frames_detected"], help="Frames requiring full color search")
    col3.metric("Target Lost Frames", stats["frames_lost"], delta="-", delta_color="inverse")

    if tracking_run.preview_frames:
        cols = st.columns(len(tracking_run.preview_frames))
        for col, img in zip(cols, tracking_run.preview_frames):
            col.image(img)

    st.success("Tracking data exported. Run: python Analyze.py --csv output/calibration_raw.csv --fps <video_fps>")
