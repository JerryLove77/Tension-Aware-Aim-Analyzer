# filepath: vision/sampler.py
# ============================================================
# 自适应颜色采样与 ROI 工具
# 零 UI 依赖，纯 CV 工具
# ============================================================

import numpy as np
import cv2


def sample_hsv_from_roi(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    h_margin: int = 10,
    s_margin: int = 50,
    v_margin: int = 50,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """从用户框选的 ROI 中采样中值 HSV，生成带容差的范围。

    Args:
        frame:   BGR 格式的完整画面。
        roi:     ``(x, y, w, h)`` 矩形区域。
        h_margin: Hue 容差（默认 ±10）。
        s_margin: Saturation 容差（默认 ±50）。
        v_margin: Value 容差（默认 ±50）。

    Returns:
        ``(hsv_lower, hsv_upper)`` 两个三元组，可直接传给 ``cv2.inRange``。
    """
    x, y, w, h = roi
    roi_pixels = frame[y : y + h, x : x + w].reshape(-1, 3)

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
