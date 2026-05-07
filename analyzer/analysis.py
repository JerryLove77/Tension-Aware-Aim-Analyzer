# filepath: analyzer/analysis.py
# ============================================================
# 运动学分析与判定模型
# 纯 NumPy/SciPy 计算层，零 UI 依赖
# 硬核模式：不插值 NaN，不设时间冗余，逐帧判定
# ============================================================

import numpy as np
from scipy.interpolate import interp1d

from .math_utils import apply_smoothing, calc_derivative

_UPSAMPLE_TARGET_FPS = 120
_MIN_CHUNK_FOR_DERIVATIVE = 5


def _upsample_chunk(
    chunk: "pd.DataFrame",
    orig_fps: float,
    target_fps: int = _UPSAMPLE_TARGET_FPS,
) -> "pd.DataFrame":
    """对帧块执行线性升采样至 target_fps。"""
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
            interp = interp1d(
                x_old, raw, kind="linear",
                bounds_error=False, fill_value="extrapolate",
            )
            data[col] = interp(x_new)

    return pd.DataFrame(data)


def compute_metrics(df: "pd.DataFrame", fps: float) -> dict[str, float]:
    """计算瞄准运动学全套指标（硬核模式）。

    Pipeline:
        1. 保留原始 NaN（不插值），NaN 帧直接按脱靶处理
        2. 按帧号间隙分割连续追踪块
        3. 块内将连续无 NaN 段升采样至 120Hz + SavGol 平滑（100ms 窗口）
        4. 欧几里得距离脱靶判定，阈值 1.45× 半尺寸，逐帧生效，无时间冗余
        5. 归一化速度/加速度偏差率（分母下限 1.0）
        6. 汇总


    Returns:
        dict: accuracy, avg_error_px, loss_count, total_off_time,
              avg_speed_mismatch, avg_accel_mismatch, ptc.
    """
    import pandas as pd

    data = df.copy()

    # ---- 1. 连续追踪块检测（仅按帧号间隙，不处理 NaN）----
    data["_gap"] = data["frame"].diff()
    data["_new_chunk"] = data["_gap"] > 1
    data["chunk_id"] = data["_new_chunk"].cumsum()

    # ---- 2. 升采样 & 平滑参数 ----
    upsample_factor = max(1, int(round(_UPSAMPLE_TARGET_FPS / fps)))
    effective_fps = fps * upsample_factor

    window_size = max(
        _MIN_CHUNK_FOR_DERIVATIVE, int(effective_fps * 0.1)
    )
    if window_size % 2 == 0:
        window_size += 1

    all_errors: list[float] = []
    all_miss: list[int] = []
    all_speed_mismatch: list[float] = []
    all_accel_mismatch: list[float] = []

    for _, chunk in data.groupby("chunk_id"):
        chunk = chunk.reset_index(drop=True)
        is_valid = chunk["ball_x"].notna() & chunk["ball_y"].notna()

        i = 0
        while i < len(chunk):
            if not is_valid.iloc[i]:
                # NaN 帧 → 直接按脱靶，补齐 upsample_factor 条记录
                n_up = upsample_factor
                all_errors.extend([float("nan")] * n_up)
                all_miss.extend([1] * n_up)
                all_speed_mismatch.extend([0.0] * n_up)
                all_accel_mismatch.extend([0.0] * n_up)
                i += 1
                continue

            # 找到连续无 NaN 段的末尾
            j = i
            while j < len(chunk) and is_valid.iloc[j]:
                j += 1

            seg = chunk.iloc[i:j].copy()

            # 升采样
            up = _upsample_chunk(seg, fps)
            n_up = len(up)

            # ---- Raw coordinates — for spatial error & hit/miss (matches VOD rendering) ----
            raw_cx = up["cross_x"].values.astype(float)
            raw_cy = up["cross_y"].values.astype(float)
            raw_bx = up["ball_x"].values.astype(float)
            raw_by = up["ball_y"].values.astype(float)
            raw_bw = up["ball_w"].values.astype(float)
            raw_bh = up["ball_h"].values.astype(float)

            # ---- Hit/miss from raw data (AABB, 1.0x threshold) ----
            dx = raw_bx - raw_cx
            dy = raw_by - raw_cy
            error_px = np.sqrt(dx**2 + dy**2)
            half_w = raw_bw / 2.0
            half_h = raw_bh / 2.0
            is_miss = ((np.abs(dx) > half_w) | (np.abs(dy) > half_h)).astype(int)

            # ---- Kinematics from smoothed data only ----
            can_smooth = n_up >= window_size
            if can_smooth:
                smooth_cx = apply_smoothing(raw_cx, window_size)
                smooth_cy = apply_smoothing(raw_cy, window_size)
                smooth_bx = apply_smoothing(raw_bx, window_size)
                smooth_by = apply_smoothing(raw_by, window_size)

                v_cx = calc_derivative(smooth_cx, effective_fps)
                v_cy = calc_derivative(smooth_cy, effective_fps)
                v_bx = calc_derivative(smooth_bx, effective_fps)
                v_by = calc_derivative(smooth_by, effective_fps)

                a_cx = calc_derivative(v_cx, effective_fps)
                a_cy = calc_derivative(v_cy, effective_fps)
                a_bx = calc_derivative(v_bx, effective_fps)
                a_by = calc_derivative(v_by, effective_fps)

                v_diff_mag = np.hypot(v_cx - v_bx, v_cy - v_by)
                v_b_mag = np.hypot(v_bx, v_by)
                speed_mismatch = v_diff_mag / np.maximum(v_b_mag, 1.0)

                a_diff_mag = np.hypot(a_cx - a_bx, a_cy - a_by)
                a_b_mag = np.hypot(a_bx, a_by)
                accel_mismatch = a_diff_mag / np.maximum(a_b_mag, 1.0)
            else:
                speed_mismatch = np.zeros(n_up, dtype=float)
                accel_mismatch = np.zeros(n_up, dtype=float)

            all_errors.extend(error_px.tolist())
            all_miss.extend(is_miss.tolist())
            all_speed_mismatch.extend(speed_mismatch.tolist())
            all_accel_mismatch.extend(accel_mismatch.tolist())

            i = j

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

    # ---- 汇总指标 ----
    accuracy = round((1.0 - miss_arr.sum() / total) * 100, 1)
    valid_errors = errors_arr[~np.isnan(errors_arr)]
    avg_error = round(float(np.mean(valid_errors)), 1) if len(valid_errors) > 0 else 0.0

    transitions = (~miss_arr[:-1]) & (miss_arr[1:])
    loss_count = int(transitions.sum())

    total_off_time = round(float(miss_arr.sum() / effective_fps), 2)

    avg_speed_mismatch = round(float(np.mean(speed_mismatch_arr)), 3)
    avg_accel_mismatch = round(float(np.mean(accel_mismatch_arr)), 3)

    ptc_val = (
        round(float(np.mean(accel_mismatch_arr[miss_arr])), 3)
        if miss_arr.any()
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "avg_error_px": avg_error,
        "loss_count": loss_count,
        "total_off_time": total_off_time,
        "avg_speed_mismatch": avg_speed_mismatch,
        "avg_accel_mismatch": avg_accel_mismatch,
        "ptc": ptc_val,
    }
