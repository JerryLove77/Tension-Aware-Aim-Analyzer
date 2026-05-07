# filepath: analyzer/smoothness.py

# ============================================================
# analyzer/smoothness.py
# 运动学分析层 — 瞄准平滑度与张力评分
# 绝对不导入 PySide6 或 app 包中的任何模块
# ============================================================

import numpy as np
from scipy.interpolate import interp1d

from .math_utils import apply_smoothing, calc_derivative

_MIN_CHUNK_FOR_DERIVATIVE = 5
_EPSILON = 1e-5
_UPSAMPLE_TARGET_FPS = 120


def _filter_short_miss_streaks(
    miss_arr: np.ndarray,
    min_streak: int = 3,
) -> np.ndarray:
    """滤除长度小于 min_streak 的连续 miss 序列（擦边命中补偿）。"""
    result = miss_arr.copy()
    padded = np.concatenate([[False], result, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for start, end in zip(starts, ends):
        if end - start < min_streak:
            result[start:end] = False
    return result


def _upsample_chunk(
    chunk: "DataFrame",
    orig_fps: float,
    target_fps: int = _UPSAMPLE_TARGET_FPS,
) -> "DataFrame":
    import pandas as pd

    factor = int(round(target_fps / orig_fps))
    n = len(chunk)
    if factor <= 1 or n < 3:
        return chunk

    x_old = np.arange(n, dtype=float)
    x_new = np.linspace(0, n - 1, n * factor)

    cols = ["cross_x", "cross_y", "ball_x", "ball_y", "ball_w", "ball_h"]
    data: dict[str, np.ndarray] = {"frame": np.arange(n * factor)}
    for col in cols:
        if col in chunk.columns:
            raw = chunk[col].values.astype(float)
            interp = interp1d(x_old, raw, kind="linear",
                              bounds_error=False, fill_value="extrapolate")
            data[col] = interp(x_new)

    return pd.DataFrame(data)


def analyze(df: "DataFrame", fps: float) -> dict[str, float]:
    """返回完整的运动学指标字典。

    Keys:
        accuracy, avg_error_px, loss_count, total_off_time,
        avg_speed_mismatch, avg_accel_mismatch, ptc.
    """
    import pandas as pd

    # 保留所有帧（含 NaN），用 chunk 内插值而非 dropna
    df_valid = df.copy()

    # Detect continuous tracking chunks (frame number gaps)
    df_valid["_gap"] = df_valid["frame"].diff()
    df_valid["_new_chunk"] = df_valid["_gap"] > 1
    df_valid["chunk_id"] = df_valid["_new_chunk"].cumsum()

    upsample_factor = max(1, int(round(_UPSAMPLE_TARGET_FPS / fps)))
    effective_fps = fps * upsample_factor

    window_size = max(_MIN_CHUNK_FOR_DERIVATIVE, int(effective_fps * 0.1))
    if window_size % 2 == 0:
        window_size += 1

    # Temporal buffer: ~50ms at original fps → scale to upsampled rate
    min_miss_streak = 3 * upsample_factor

    all_errors: list[float] = []
    all_miss: list[int] = []
    all_speed_mismatch: list[float] = []
    all_accel_mismatch: list[float] = []

    for _, chunk in df_valid.groupby("chunk_id"):
        chunk = chunk.reset_index(drop=True)

        # ---- NaN 线性插值：恢复短暂追踪丢失 ----
        ball_cols = ["ball_x", "ball_y", "ball_w", "ball_h"]
        chunk[ball_cols] = chunk[ball_cols].interpolate(
            method="linear", limit_direction="both"
        )

        # ---- CSRT 延迟补偿：ball 位置前移 1 帧 ----
        chunk[["ball_x", "ball_y"]] = chunk[["ball_x", "ball_y"]].shift(-1)
        chunk[["ball_x", "ball_y"]] = chunk[["ball_x", "ball_y"]].interpolate(
            method="linear", limit_direction="both"
        )

        chunk = _upsample_chunk(chunk, fps)
        n = len(chunk)

        cross_x = chunk["cross_x"].values.astype(float)
        cross_y = chunk["cross_y"].values.astype(float)
        ball_x = chunk["ball_x"].values.astype(float)
        ball_y = chunk["ball_y"].values.astype(float)
        ball_w = chunk["ball_w"].values.astype(float)
        ball_h = chunk["ball_h"].values.astype(float)

        can_smooth = n >= window_size

        if can_smooth:
            smooth_cx = apply_smoothing(cross_x, window_size)
            smooth_cy = apply_smoothing(cross_y, window_size)
            smooth_tx = apply_smoothing(ball_x, window_size)
            smooth_ty = apply_smoothing(ball_y, window_size)
            smooth_w = apply_smoothing(ball_w, window_size)
            smooth_h = apply_smoothing(ball_h, window_size)
        else:
            smooth_cx, smooth_cy = cross_x, cross_y
            smooth_tx, smooth_ty = ball_x, ball_y
            smooth_w, smooth_h = ball_w, ball_h

        error_px = np.hypot(smooth_tx - smooth_cx, smooth_ty - smooth_cy)
        # Dynamic hitbox: error > (smoothed half-width × 1.35)
        is_miss = (error_px > (smooth_w / 2.0 * 1.35)).astype(int)

        # Temporal buffer: 短于 ~50ms 的脱靶不计数
        is_miss = _filter_short_miss_streaks(
            is_miss.astype(bool), min_streak=min_miss_streak
        ).astype(int)

        if can_smooth:
            v_cx = calc_derivative(smooth_cx, effective_fps)
            v_cy = calc_derivative(smooth_cy, effective_fps)
            v_tx = calc_derivative(smooth_tx, effective_fps)
            v_ty = calc_derivative(smooth_ty, effective_fps)

            a_cx = calc_derivative(v_cx, effective_fps)
            a_cy = calc_derivative(v_cy, effective_fps)
            a_tx = calc_derivative(v_tx, effective_fps)
            a_ty = calc_derivative(v_ty, effective_fps)

            v_diff_mag = np.hypot(v_cx - v_tx, v_cy - v_ty)
            v_t_mag = np.hypot(v_tx, v_ty)
            speed_mismatch = v_diff_mag / np.maximum(v_t_mag, _EPSILON)

            a_diff_mag = np.hypot(a_cx - a_tx, a_cy - a_ty)
            a_t_mag = np.hypot(a_tx, a_ty)
            accel_mismatch = a_diff_mag / np.maximum(a_t_mag, _EPSILON)
        else:
            speed_mismatch = np.zeros(n, dtype=float)
            accel_mismatch = np.zeros(n, dtype=float)

        all_errors.extend(error_px.tolist())
        all_miss.extend(is_miss.tolist())
        all_speed_mismatch.extend(speed_mismatch.tolist())
        all_accel_mismatch.extend(accel_mismatch.tolist())

    total = len(all_errors)
    if total == 0:
        return {
            "accuracy": 0.0,
            "avg_error_px": 0.0,
            "loss_count": 0,
            "total_off_time": 0.0,
            "avg_speed_mismatch": 0.0,
            "avg_accel_mismatch": 0.0,
            "ptc": 0.0,
        }

    errors_arr = np.array(all_errors)
    miss_arr = np.array(all_miss, dtype=bool)
    speed_mismatch_arr = np.array(all_speed_mismatch)
    accel_mismatch_arr = np.array(all_accel_mismatch)

    accuracy = round((1.0 - miss_arr.sum() / total) * 100, 1)
    avg_error = round(float(np.mean(errors_arr)), 1)

    # Loss count = transitions from tracking → miss (post-filtered)
    transitions = (~miss_arr[:-1]) & (miss_arr[1:])
    loss_count = int(transitions.sum())

    # Total off time = miss frames / effective_fps (upsampled rate)
    total_off_time = round(float(miss_arr.sum() / effective_fps), 2)

    avg_speed_mismatch = round(float(np.mean(speed_mismatch_arr)), 3)
    avg_accel_mismatch = round(float(np.mean(accel_mismatch_arr)), 3)

    if miss_arr.any():
        ptc_val = round(float(np.mean(accel_mismatch_arr[miss_arr])), 3)
    else:
        ptc_val = 0.0

    return {
        "accuracy": accuracy,
        "avg_error_px": avg_error,
        "loss_count": loss_count,
        "total_off_time": total_off_time,
        "avg_speed_mismatch": avg_speed_mismatch,
        "avg_accel_mismatch": avg_accel_mismatch,
        "ptc": ptc_val,
    }
