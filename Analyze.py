"""
kovaak_tracker/analyze.py
==========================
Step 2: Physics Analysis (V5 - Tension Vector Edition)
Calculates Speed Matching accuracy via the J/E (Jitter/Error) Ratio.
"""

import argparse
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter

def run_tension_analysis(df, fps):
    """Calculates TBR (Tension Balance Ratio) to diagnose aim tension."""
    window_sz = max(5, int(fps * 0.1)) 
    all_E, all_J = [], []
    total_overshoot = 0
    
    for _, group in df.groupby('chunk_id'):
        if len(group) < window_sz: continue
            
        # Smoothing trajectory to remove sensor noise
        cx, cy = savgol_filter(group["cross_x"].values, window_sz, 3), savgol_filter(group["cross_y"].values, window_sz, 3)
        bx, by = savgol_filter(group["ball_x"].values, window_sz, 3), savgol_filter(group["ball_y"].values, window_sz, 3)

        # Vector E: Spatial Error
        dx, dy = bx - cx, by - cy
        dist_E = np.hypot(dx, dy)
        all_E.extend(dist_E)

        # Vector J: Relative Acceleration (Jitter)
        vx, vy = np.diff(dx) * fps, np.diff(dy) * fps
        speed = np.hypot(vx, vy)
        accel_J = np.abs(np.diff(speed)) * fps
        all_J.extend(np.append(accel_J, accel_J[-1]))

    # Normalization (Baseline: 20px error ~ 15k px/s^2 jitter)
    norm_E = np.mean(all_E) / 20.0
    norm_J = np.mean(all_J) / 15000.0
    tbr = norm_J / (norm_E + 1e-6)

    # Automated Diagnosis based on MattyOW's theory
    if tbr > 1.8: diagnosis = "Over-tension: High micro-jitter relative to error."
    elif tbr < 0.6: diagnosis = "Under-tension: Lagging response, low corrective force."
    else: diagnosis = "Good Speed Matching: Balanced muscle tension."

    return {
        "avg_error_px": round(float(np.mean(all_E)), 2),
        "jitter_rate": round(float(np.mean(all_J)), 1),
        "tension_ratio": round(float(tbr), 2),
        "tension_score": round(100 * (math.e ** (-0.5 * math.sqrt(norm_E**2 + norm_J**2))), 1),
        "diagnosis": diagnosis
    }
# [Omitted boilerplate main() and summary formatting]