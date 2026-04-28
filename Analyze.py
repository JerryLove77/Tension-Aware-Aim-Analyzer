"""
kovaak_tracker/analyze.py
==========================
V8: Vectorized Kinematics & Dual Mismatch Engine
Architecture:
- Stage 1: Full Kinematics Extraction (Vectorized)
- Stage 2: Masked Physics Evaluation (Miss Mask Logic)
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter

# ══════════════════════════════════════════
#  Basic Utility Functions
# ══════════════════════════════════════════

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['is_valid'] = df['ball_x'].notna()
    valid_df = df[df['is_valid']].copy().sort_values("frame")
    
    if valid_df.empty:
        raise ValueError("Invalid video data: No target detected!")
        
    valid_df['frame_diff'] = valid_df['frame'].diff()
    valid_df['chunk_id'] = (valid_df['frame_diff'] > 2).cumsum()
    valid_df = valid_df.fillna({'frame_diff': 1})
    return valid_df.reset_index(drop=True)

def apply_smoothing(series: np.ndarray, window_length: int, polyorder: int = 3) -> np.ndarray:
    """Applies Savitzky-Golay filter to smooth trajectory data."""
    if window_length % 2 == 0: window_length += 1
    if len(series) < window_length: return series
    return savgol_filter(series, window_length, polyorder)

def calc_derivative(series: np.ndarray, fps: float) -> np.ndarray:
    """Calculates the derivative and pads the last element to maintain array length."""
    diffs = np.diff(series) * fps
    return np.append(diffs, diffs[-1])

# ══════════════════════════════════════════
#  Stage 1: Kinematics Extraction
# ══════════════════════════════════════════

def extract_kinematics(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    window_sz = max(5, int(fps * 0.1)) 
    processed_chunks = []
    
    for chunk_id, group in df.groupby('chunk_id'):
        if len(group) < window_sz: continue
            
        # 1. Trajectory Smoothing
        cx, cy = apply_smoothing(group["cross_x"].values, window_sz), apply_smoothing(group["cross_y"].values, window_sz)
        bx, by = apply_smoothing(group["ball_x"].values, window_sz), apply_smoothing(group["ball_y"].values, window_sz)
        bw, bh = group["ball_w"].values, group["ball_h"].values

        # 2. Spatial Error & Hit Detection (Build Miss Mask)
        dx, dy = bx - cx, by - cy
        error_px = np.hypot(dx, dy)
        # Rectangular Hitbox: Miss (1.0) if outside edges, otherwise Hit (0.0)
        is_miss = ((np.abs(dx) > (bw / 2 * 1.35)) | (np.abs(dy) > (bh / 2 * 1.35))).astype(float)

        # 3. Target Kinematics
        v_tx, v_ty = calc_derivative(bx, fps), calc_derivative(by, fps)
        a_tx, a_ty = calc_derivative(v_tx, fps), calc_derivative(v_ty, fps)

        # 4. Crosshair Kinematics
        v_cx, v_cy = calc_derivative(cx, fps), calc_derivative(cy, fps)
        a_cx, a_cy = calc_derivative(v_cx, fps), calc_derivative(v_cy, fps)
        
        # 5. Mismatch Vectors (Magnitude of differences)
        v_rel = np.hypot(v_cx - v_tx, v_cy - v_ty) # Speed Mismatch
        a_rel = np.hypot(a_cx - a_tx, a_cy - a_ty) # Acceleration Mismatch

        v_rel = np.hypot(v_cx - v_tx, v_cy - v_ty) # 速度失配
        a_rel = np.hypot(a_cx - a_tx, a_cy - a_ty) # 加速度失配

        processed_chunks.append(pd.DataFrame({
            'frame': group['frame'].values, 'time_s': group['time_s'].values,
            'chunk_id': chunk_id, 'error_px': error_px, 'is_miss': is_miss,
            'speed_t': np.hypot(v_tx, v_ty), 'accel_t': np.hypot(a_tx, a_ty),
            'speed_c': np.hypot(v_cx, v_cy), 'accel_c': np.hypot(a_cx, a_cy),
            'v_rel': v_rel, 'a_rel': a_rel,
            # [新增] 为 Dashboard 录像回放保留原始坐标数据
            'cross_x': group['cross_x'].values, 'cross_y': group['cross_y'].values,
            'ball_x': group['ball_x'].values, 'ball_y': group['ball_y'].values,
            'ball_w': bw, 'ball_h': bh
        }))

    return pd.concat(processed_chunks, ignore_index=True)

# ══════════════════════════════════════════
#  Stage 2: Vectorized Evaluation
# ══════════════════════════════════════════

def evaluate_mechanics(kdf: pd.DataFrame) -> dict:
    # Extract Feature Vectors
    M = kdf['is_miss'].values      # Miss Mask Vector
    V_rel = kdf['v_rel'].values    # Speed Mismatch Vector
    A_rel = kdf['a_rel'].values    # Accel Mismatch Vector
    E = kdf['error_px'].values     # Error Vector
    
    total_miss_frames = np.sum(M)
    total_frames = len(kdf)
    accuracy = (1 - (total_miss_frames / total_frames)) * 100 if total_frames > 0 else 0

    if total_miss_frames > 0:
        # Use vector dot product for Masked average calculation
        # Only evaluate physics features during miss frames (where M == 1)
        mean_v_mismatch = np.dot(M, V_rel) / total_miss_frames
        mean_a_mismatch = np.dot(M, A_rel) / total_miss_frames
        mean_e_mismatch = np.dot(M, E) / total_miss_frames
        
        # PTC: Acceleration Mismatch / Spatial Error (Measures Arm/Spring Stiffness)
        ptc = mean_a_mismatch / max(mean_e_mismatch, 1.0)
    else:
        mean_v_mismatch = mean_a_mismatch = ptc = 0.0

    # Calculate loss events for dashboard compatibility
    loss_events = (kdf['is_miss'].diff() > 0).sum()
    total_off_time = total_miss_frames / (fps := 1 / kdf['time_s'].diff().mean() if total_frames > 0 else 60)

    return {
        "accuracy": round(accuracy, 1),
        "speed_mismatch": round(mean_v_mismatch, 1),
        "accel_mismatch": round(mean_a_mismatch, 0),
        "ptc": round(ptc, 1),
        "avg_error_px": round(np.mean(E), 2),
        "loss_count": loss_events,
        "total_off_time": round(total_off_time, 2)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--fps", type=float, default=60)
    args = parser.parse_args()

    df = load_data(args.csv)
    kdf = extract_kinematics(df, args.fps)
    res = evaluate_mechanics(kdf)
    
    print("\n" + "═" * 70)
    print(f"  KovaaK Vectorized Physics Measurement Report (V8 Edition)")
    print("═" * 70)
    print(f"  Accuracy (On-Target)    : {res['accuracy']}%")
    print(f"  Speed Mismatch (ΔV)     : {res['speed_mismatch']} px/s")
    print(f"  Tension Mismatch (ΔA)   : {res['accel_mismatch']} px/s²")
    print(f"  Pure Tension Coeff (PTC): {res['ptc']} Hz² (Stiffness)")
    print(f"  Global Avg Spatial Error: {res['avg_error_px']} px")
    print("═" * 70 + "\n")

    # Export Data
    out_dir = Path("output")
    kdf['on_target'] = 1 - kdf['is_miss'] # Alias for Dashboard
    kdf.to_csv(out_dir / "frame_errors.csv", index=False)
    
    export_format = {
        "tension": {
            "avg_error_px": float(res["avg_error_px"]),
            "speed_mismatch": float(res["speed_mismatch"]),
            "accel_mismatch": float(res["accel_mismatch"]),
            "ptc": float(res["ptc"])
        },
        "loss": {
            "on_target_pct": float(res["accuracy"]),
            "loss_count": int(res["loss_count"]),
            "total_off_time": float(res["total_off_time"])
        }
    }
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(export_format, f)


if __name__ == "__main__":
    main()