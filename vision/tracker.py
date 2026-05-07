# filepath: vision/tracker.py
# ============================================================
# 追踪引擎 — 混合颜色检测 + CSRT 追踪管线
# 接收外部传入的 HSV 范围和帧范围，输出统一列名的 DataFrame
# 零 UI 依赖
# ============================================================

from __future__ import annotations

import os
from typing import Any, Callable

import cv2
import numpy as np

from .vision import detect_point_by_color, refine_local_bbox


def run_tracking_analysis(
    video_path: str,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
    start_frame: int = 0,
    end_frame: int | None = None,
    progress_callback: Callable[[float], None] | None = None,
    min_area: int = 50,
    max_area: int = 5000,
) -> "pd.DataFrame":
    """对视频执行混合目标追踪（颜色检测 → CSRT 追踪 → 回退）。

    Pipeline 逻辑:
        1. 若 CSRT 追踪器存活 → 调用 ``tracker.update()``
        2. 若追踪器丢失或不存在 → HSV 颜色检测 → 找到最佳候选 → 初始化 CSRT
        3. 重复直到帧结束

    Args:
        video_path:  视频文件路径。
        hsv_lower:  HSV 颜色下界 (H, S, V)。
        hsv_upper:  HSV 颜色上界 (H, S, V)。
        start_frame: 起始帧号（默认 0）。
        end_frame:   结束帧号（不含，默认 None = 视频末尾）。
        progress_callback: 可选进度回调，``(progress_fraction)``。
        min_area:   有效目标轮廓的最小面积（像素）。
        max_area:   有效目标轮廓的最大面积（像素）。

    Returns:
        ``pandas.DataFrame``，列:
            frame, time_s, ball_x, ball_y, ball_w, ball_h, cross_x, cross_y
        追踪丢失时 ball_x / ball_y / ball_w / ball_h 记为 ``np.nan``。
    """
    import pandas as pd

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    if end_frame <= start_frame:
        cap.release()
        return pd.DataFrame(
            columns=[
                "frame", "time_s", "ball_x", "ball_y",
                "ball_w", "ball_h", "cross_x", "cross_y",
            ]
        )

    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    cx, cy = width // 2, height // 2  # 准星 = 画面中心
    records: list[dict[str, Any]] = []
    tracker: cv2.TrackerCSRT | None = None
    prev_w: int = 40   # 上一帧目标宽度（用于 refine_local_bbox 的 ROI 尺寸）
    prev_h: int = 40   # 上一帧目标高度

    total_in_range = end_frame - start_frame
    progress_interval = max(1, total_in_range // 100)

    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        bx = float("nan")
        by = float("nan")
        bw = float("nan")
        bh = float("nan")

        # ---------------------------------------------------------------
        # Phase A: CSRT 粗定位 → 局部 Mask 紧密扣边
        # ---------------------------------------------------------------
        if tracker is not None:
            success, bbox = tracker.update(frame)
            if success:
                _x, _y, _w, _h = [int(v) for v in bbox]
                tx, ty = _x + _w // 2, _y + _h // 2
                # 局部 Mask 精炼：ROI 开窗 → HSV 二值 → 准星剔除 → 紧贴轮廓
                bx, by, bw, bh = refine_local_bbox(
                    frame, tx, ty, hsv_lower, hsv_upper, cx, cy,
                    prev_w=prev_w, prev_h=prev_h,
                )
                prev_w, prev_h = bw, bh  # 更新开窗尺寸供下一帧使用
            else:
                tracker = None  # 追踪丢失 → 回退到颜色检测

        # ---------------------------------------------------------------
        # Phase B: 颜色检测 → 重新初始化 CSRT
        # ---------------------------------------------------------------
        if tracker is None:
            bbox = detect_point_by_color(
                frame, hsv_lower, hsv_upper, cx, cy,
                min_area=min_area, max_area=max_area,
            )
            if bbox is not None:
                _x, _y, _w, _h = bbox
                prev_w, prev_h = _w, _h  # 初始化局部开窗尺寸
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (_x, _y, _w, _h))
                bx, by = _x + _w // 2, _y + _h // 2
                bw, bh = _w, _h

        # ---------------------------------------------------------------
        # 统一记录
        # ---------------------------------------------------------------
        ts = frame_idx / fps if fps > 0 else 0.0
        records.append(
            {
                "frame": frame_idx,
                "time_s": ts,
                "ball_x": bx,
                "ball_y": by,
                "ball_w": bw,
                "ball_h": bh,
                "cross_x": cx,
                "cross_y": cy,
            }
        )

        # 进度回调
        frames_done = frame_idx - start_frame + 1
        if progress_callback and frames_done % progress_interval == 0:
            progress_callback(frames_done / total_in_range)

    cap.release()

    if progress_callback:
        progress_callback(1.0)

    return pd.DataFrame(records)
