"""
kovaak_tracker/app.py
======================
V8: Tracking-by-Detection Architecture (CSRT Engine)
- Step 1: Detect object using strict color/morphology parameters.
- Step 2: Once found, hand over to cv2.TrackerCSRT for high-speed, robust tracking.
- Step 3: If tracker loses confidence, instantly fallback to Detection.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import tempfile
from pathlib import Path

st.set_page_config(page_title="KovaaK Tracker · Calibration", layout="wide")
st.title("KovaaK Tracking Analyzer (Pro Engine)")
st.caption("Powered by Tracking-by-Detection (CSRT) Architecture")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════
#  CV Engine Components
# ══════════════════════════════════════════

def get_hsv_range(bgr_color):
    """Generates an adaptive HSV range based on the sampled BGR color."""
    hsv = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    
    if v < 40:  
        return np.array([0, 0, 0]), np.array([179, 255, v + 25])
    elif s < 30 and v > 200: 
        return np.array([0, 0, max(0, v - 50)]), np.array([179, 50, 255])
    else: 
        return np.array([max(0, h-10), max(0, s-50), max(0, v-50)]), \
               np.array([min(179, h+10), min(255, s+50), min(255, v+50)])

def detect_ball_by_color(frame, hsv_lo, hsv_hi):
    """Fallback Detection Engine (Used only for initial lock or recovery)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None, None
    
    h_img, w_img = frame.shape[:2]
    center_x, center_y = w_img // 2, h_img // 2
    max_valid_area = w_img * h_img * 0.05 
    
    best_pos, best_w, best_h = None, None, None
    min_dist = float('inf')
    
    for c in contours:
        area = cv2.contourArea(c)
        if 20 < area < max_valid_area:
            _, _, w, h = cv2.boundingRect(c)
            if h == 0: continue
            
            aspect_ratio = w / float(h)
            if aspect_ratio > 1.3: continue 
                
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Ignore UI elements near the top
                if cy < h_img * 0.12 and w > 60: continue
                
                dist = (cx - center_x)**2 + (cy - center_y)**2
                if dist < min_dist:
                    min_dist = dist
                    best_pos = (cx, cy)
                    best_w, best_h = w, h
                    
    return best_pos, best_w, best_h

def get_tracker():
    """Initializes the CSRT Tracker (or fallback to KCF)"""
    try:
        return cv2.TrackerCSRT_create()
    except AttributeError:
        st.warning("CSRT not found, falling back to KCF. Please install opencv-contrib-python.")
        return cv2.TrackerKCF_create()

def frame_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ══════════════════════════════════════════
#  Streamlit UI & Pipeline
# ══════════════════════════════════════════

video_file = st.file_uploader("Support mp4 / avi / mov / webm", type=["mp4", "avi", "mov", "webm"])
if video_file is None: st.stop()

file_extension = Path(video_file.name).suffix
with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
    tmp.write(video_file.read())
    video_path = tmp.name

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
st.success(f"Video Loaded: {width}x{height} @ {fps:.1f}fps")
cap.release()

st.header("1. Trim Valid Video Clip")
start_frame, end_frame = st.slider("Select Valid Video Range", 0, total, (0, total))
process_length = end_frame - start_frame

st.header("2. Extract Target Color")
template_frame_idx = st.slider("Slide to find the target ball", start_frame, max(start_frame, end_frame - 1), start_frame)

cap2 = cv2.VideoCapture(video_path)
cap2.set(cv2.CAP_PROP_POS_FRAMES, template_frame_idx)
ret2, selected_frame = cap2.read()
cap2.release()

st.image(frame_to_rgb(selected_frame), use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
x1 = col1.number_input("Left x1", 0, width,  width//2 - 10)
y1 = col2.number_input("Top y1", 0, height, height//2 - 10)
x2 = col3.number_input("Right x2", 0, width,  width//2 + 10)
y2 = col4.number_input("Bottom y2", 0, height, height//2 + 10)

if x2 <= x1 or y2 <= y1: st.stop()

template = selected_frame[int(y1):int(y2), int(x1):int(x2)]
median_color = np.median(template.reshape(-1, 3), axis=0).astype(np.uint8)
ball_bgr = median_color.tolist()
b_lo, b_hi = get_hsv_range(ball_bgr)

preview = selected_frame.copy()
cv2.rectangle(preview, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
st.image(frame_to_rgb(preview), caption=f"Extracted RGB: {ball_bgr[::-1]}")

st.header("3. Start Analysis (Tracking-by-Detection)")

if st.button("▶ Ignite Physics Engine", type="primary"):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    results, preview_frames = [], []
    cross_pos = (width // 2, height // 2)
    progress = st.progress(0, text="Initializing Tracking Engine...")

    # 🚀 The Tracking-by-Detection State Machine
    tracking_active = False
    tracker = None
    
    # Engine Statistics
    stats = {"frames_detected": 0, "frames_tracked": 0, "frames_lost": 0}

    for i in range(process_length):
        ret, frame = cap.read()
        if not ret: break

        vis = frame.copy()
        absolute_frame_idx = start_frame + i 
        
        cv2.drawMarker(vis, cross_pos, (0, 180, 255), cv2.MARKER_CROSS, 18, 2)
        
        ball_pos, ball_w, ball_h = None, None, None

        if not tracking_active:
            # ── STATE 1: DETECTION MODE (Heavy Search) ──
            b_pos, b_w, b_h = detect_ball_by_color(frame, b_lo, b_hi)
            
            if b_pos:
                # Target acquired! Hand over to Tracker.
                cx, cy = b_pos
                bbox = (int(cx - b_w/2), int(cy - b_h/2), int(b_w), int(b_h))
                
                tracker = get_tracker()
                tracker.init(frame, bbox)
                tracking_active = True
                
                ball_pos, ball_w, ball_h = b_pos, b_w, b_h
                stats["frames_detected"] += 1
                box_color = (0, 255, 255) # Yellow box indicates "Detection Mode"
            else:
                stats["frames_lost"] += 1
        else:
            # ── STATE 2: TRACKING MODE (High Speed) ──
            success, bbox = tracker.update(frame)
            
            if success:
                # Tracking holds!
                x, y, w, h = [int(v) for v in bbox]
                ball_pos = (int(x + w/2), int(y + h/2))
                ball_w, ball_h = w, h
                stats["frames_tracked"] += 1
                box_color = (0, 220, 80) # Green box indicates "Tracking Mode"
            else:
                # Tracker lost confidence! Fallback immediately.
                tracking_active = False
                tracker = None
                stats["frames_lost"] += 1

        # Draw visuals if target exists
        if ball_pos:
            cx, cy = ball_pos
            top_left = (int(cx - ball_w / 2), int(cy - ball_h / 2))
            bottom_right = (int(cx + ball_w / 2), int(cy + ball_h / 2))
            
            cv2.rectangle(vis, top_left, bottom_right, box_color, 2)
            cv2.circle(vis, (cx, cy), 2, box_color, -1)
            cv2.line(vis, ball_pos, cross_pos, (255, 200, 0), 1)

        results.append({
            "frame": absolute_frame_idx, 
            "time_s": round(absolute_frame_idx / fps, 3),
            "ball_x": ball_pos[0] if ball_pos else None,
            "ball_y": ball_pos[1] if ball_pos else None,
            "ball_w": ball_w if ball_pos else None,  
            "ball_h": ball_h if ball_pos else None,
            "cross_x": cross_pos[0], "cross_y": cross_pos[1],
        })

        if i % 20 == 0: preview_frames.append(frame_to_rgb(vis))
        progress.progress((i + 1) / process_length)

    cap.release()
    progress.empty()

    # Engine Performance Report
    st.subheader("🏎️ Tracking Engine Performance")
    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("High-Speed Tracked Frames", stats["frames_tracked"], help="Green boxes: Frames handled by CSRT")
    col_e2.metric("Color Detected Frames", stats["frames_detected"], help="Yellow boxes: Frames requiring full search")
    col_e3.metric("Target Lost Frames", stats["frames_lost"], delta="-", delta_color="inverse")
    
    cols = st.columns(len(preview_frames))
    for col, img in zip(cols, preview_frames): col.image(img)

    config = {
        "ball_bgr": ball_bgr, "ball_hsv_lo": b_lo.tolist(), "ball_hsv_hi": b_hi.tolist(),
        "crosshair_mode": "center", "fps": fps, "resolution": [width, height],
    }
    with open(OUTPUT_DIR / "calib_config.json", "w") as f: json.dump(config, f)
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "calibration_raw.csv", index=False)

    st.success("✨ High-Fidelity Data Extracted! Run Analyze.py in terminal to generate the PTC report.")