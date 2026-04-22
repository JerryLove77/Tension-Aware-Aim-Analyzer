"""
kovaak_tracker/app.py
======================
Step 1: Calibration & Feature Extraction (V5)
Functions: Video trimming, color sampling, and initial computer vision pass.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import tempfile
from pathlib import Path

st.set_page_config(page_title="KovaaK Tracker - Calibrate", layout="wide")
st.title("🎯 KovaaK Vision Tracker")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

def get_hsv_range(bgr_color):
    """Generates HSV bounds with adaptive tolerance based on brightness."""
    hsv = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    
    if v < 40:  # Dark targets (e.g., Black Smoothbot)
        return np.array([0, 0, 0]), np.array([179, 255, v + 25])
    elif s < 30 and v > 200: # White targets
        return np.array([0, 0, max(0, v - 50)]), np.array([179, 50, 255])
    else: # Vivid colors
        return np.array([max(0, h-10), max(0, s-50), max(0, v-50)]), \
               np.array([min(179, h+10), min(255, s+50), min(255, v+50)])

def detect_ball_by_color(frame, hsv_lo, hsv_hi):
    """Detects target using color masking and geometry-based UI filtering."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
    
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
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
            
            # Feature Filtering: Targets are roughly square or tall (AR < 1.3). 
            # Horizontal bars (UI) are discarded.
            aspect_ratio = w / float(h)
            if aspect_ratio > 1.3: continue 
                
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                
                # Exclude static UI elements in the top 12% of the screen
                if cy < h_img * 0.12 and w > 60: continue
                
                dist = (cx - center_x)**2 + (cy - center_y)**2
                if dist < min_dist:
                    min_dist, best_pos, best_w, best_h = dist, (cx, cy), w, h
                    
    return best_pos, best_w, best_h

# UI/Workflow logic for GitHub users
# [Omitted streamlit UI boilerplate for brevity, same as V4 logic provided previously]