# filepath: vision/video_reader.py

# ============================================================
# vision/video_reader.py
# 视觉感知层 — 视频读取与混合目标追踪 (Color + CSRT)
# 绝对不导入 PySide6 或 app 包中的任何模块
# ============================================================

from __future__ import annotations

import os
from types import TracebackType
from typing import Any, Callable, Generator, Self

import cv2
import numpy as np


# ==================================================================
# 自适应 HSV 采样工具
# ==================================================================

def sample_hsv_from_roi(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    h_margin: int = 15,
    s_margin: int = 30,
    v_margin: int = 40,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """在用户框选的 ROI 内计算中值 HSV，并生成带容差的上下界。

    Args:
        frame:   BGR 格式的完整画面。
        roi:     ``(x, y, w, h)`` 矩形区域。
        h_margin: Hue 容差（默认 ±15）。
        s_margin: Saturation 容差（默认 ±30）。
        v_margin: Value 容差（默认 ±40）。

    Returns:
        ``(hsv_lower, hsv_upper)`` 两个三元组，可直接传给 ``cv2.inRange``。
    """
    x, y, w, h = roi
    roi_pixels = frame[y : y + h, x : x + w].reshape(-1, 3)

    # 用中值抵抗少量背景噪点
    median_bgr = np.median(roi_pixels, axis=0).astype(np.uint8).reshape(1, 1, 3)
    median_hsv = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2HSV)[0, 0]
    h0, s0, v0 = int(median_hsv[0]), int(median_hsv[1]), int(median_hsv[2])

    hsv_lower = (
        max(0, h0 - h_margin),
        max(0, s0 - s_margin),
        max(0, v0 - v_margin),
    )
    hsv_upper = (
        min(179, h0 + h_margin),
        min(255, s0 + s_margin),
        min(255, v0 + v_margin),
    )
    return hsv_lower, hsv_upper


# ==================================================================
# 视频处理器
# ==================================================================

class VideoProcessor:
    """使用 OpenCV 读取视频，并提供逐帧目标追踪能力。"""

    def __init__(self, path: str) -> None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"视频文件不存在: {path}")

        self._path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {path}")

        self._total_frames: int = int(
            self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        self._fps: float = self._cap.get(cv2.CAP_PROP_FPS)
        self._width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ------------------------------------------------------------------
    # 公开属性
    # ------------------------------------------------------------------

    @property
    def path(self) -> str:
        return self._path

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def duration_sec(self) -> float:
        if self._fps <= 0:
            return 0.0
        return self._total_frames / self._fps

    # ------------------------------------------------------------------
    # 核心追踪方法
    # ------------------------------------------------------------------

    def run_tracking(
        self,
        progress_callback: Callable[[float, Any], None],
        hsv_lower: tuple[int, int, int],
        hsv_upper: tuple[int, int, int],
        start_frame: int = 0,
        end_frame: int | None = None,
        min_area: int = 50,
        max_area: int = 5000,
    ) -> "DataFrame":
        """对指定帧范围执行混合追踪 (颜色检测 + CSRT)。

        Args:
            progress_callback: ``(progress_fraction, auxiliary)``
                每处理约 1% 的帧时调用一次；auxiliary 始终为 None。
            hsv_lower: HSV 颜色下界 (H, S, V)。
            hsv_upper: HSV 颜色上界 (H, S, V)。
            start_frame: 起始帧号（默认 0）。
            end_frame:   结束帧号（不含，默认 None = 视频末尾）。
            min_area: 有效目标轮廓的最小面积（像素）。
            max_area: 有效目标轮廓的最大面积（像素）。

        Returns:
            ``pandas.DataFrame`` (列: frame, time_s, cross_x, cross_y,
            ball_x, ball_y, ball_w, ball_h, tracked)。
            追踪丢失时 ball_x / ball_y / ball_w / ball_h 记为 NaN。
        """
        import pandas as pd

        if end_frame is None or end_frame > self._total_frames:
            end_frame = self._total_frames

        total_in_range = end_frame - start_frame
        if total_in_range <= 0:
            return pd.DataFrame(
                columns=[
                    "frame", "time_s", "cross_x", "cross_y",
                    "ball_x", "ball_y", "ball_w", "ball_h", "tracked",
                ]
            )

        # 跳转到起始帧
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        cx, cy = self._width // 2, self._height // 2  # 准星 = 画面中心
        records: list[dict[str, Any]] = []
        tracker: cv2.TrackerCSRT | None = None

        progress_interval = max(1, total_in_range // 100)

        for frame_idx in range(start_frame, end_frame):
            ret, frame = self._cap.read()
            if not ret:
                break

            tracked_this = False
            bx = float("nan")
            by = float("nan")
            bw = float("nan")
            bh = float("nan")

            # -------------------------------------------------------
            # Phase A: CSRT 追踪器启用 → 直接更新
            # -------------------------------------------------------
            if tracker is not None:
                success, bbox = tracker.update(frame)
                if success:
                    _x, _y, _w, _h = [int(v) for v in bbox]
                    bx, by = _x + _w // 2, _y + _h // 2
                    bw, bh = _w, _h
                    tracked_this = True
                else:
                    tracker = None

            # -------------------------------------------------------
            # Phase B: 无追踪器 → HSV 颜色检测 → 初始化 CSRT
            # -------------------------------------------------------
            if tracker is None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                candidates: list[tuple[float, int, int, Any]] = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < min_area or area > max_area:
                        continue
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    centroid_x = int(M["m10"] / M["m00"])
                    centroid_y = int(M["m01"] / M["m00"])
                    dist = np.hypot(centroid_x - cx, centroid_y - cy)
                    candidates.append((dist, centroid_x, centroid_y, cnt))

                if candidates:
                    candidates.sort(key=lambda x: x[0])
                    _, cx_, cy_, best_cnt = candidates[0]
                    _x, _y, _w, _h = cv2.boundingRect(best_cnt)

                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, (_x, _y, _w, _h))

                    bx, by = cx_, cy_
                    bw, bh = _w, _h
                    tracked_this = True

            # -------------------------------------------------------
            # 记录当前帧
            # -------------------------------------------------------
            ts = frame_idx / self._fps
            records.append(
                {
                    "frame": frame_idx,
                    "time_s": ts,
                    "cross_x": cx,
                    "cross_y": cy,
                    "ball_x": bx,
                    "ball_y": by,
                    "ball_w": bw,
                    "ball_h": bh,
                    "tracked": tracked_this,
                }
            )

            # 进度回调（基于 segment 长度）
            frames_done = frame_idx - start_frame + 1
            if frames_done % progress_interval == 0:
                progress_callback(frames_done / total_in_range, None)

        progress_callback(1.0, None)
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 逐帧生成器（兼容旧接口）
    # ------------------------------------------------------------------

    def extract_frames(
        self,
        step: int = 1,
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """按顺序生成 ``(index, frame_bgr)`` 元组。"""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if idx % step == 0:
                yield idx, frame
            idx += 1

    # ------------------------------------------------------------------
    # 资源释放 & 上下文管理器
    # ------------------------------------------------------------------

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None  # type: ignore[assignment]

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()
