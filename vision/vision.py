# filepath: vision/vision.py
# ============================================================
# 核心视觉识别模块
# 颜色检测 + HSV 采样，零 UI 依赖
# ============================================================

import numpy as np
import cv2


def detect_point_by_color(
    frame: np.ndarray,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
    cx: int,
    cy: int,
    min_area: int = 50,
    max_area: int = 5000,
) -> tuple[int, int, int, int] | None:
    """基于 HSV 颜色检测目标，返回 ``(x, y, w, h)`` 或 ``None``。

    在 ``findContours`` 之前，将 mask 正中央 30×30 区域强行涂黑，
    从物理上杜绝追踪器在初始化时被屏幕中心的准星绑架。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # ---- 屏蔽中心 30×30 准星区域 ----
    h, w = mask.shape
    x1 = max(0, cx - 15)
    y1 = max(0, cy - 15)
    x2 = min(w, cx + 15)
    y2 = min(h, cy + 15)
    mask[y1:y2, x1:x2] = 0

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates: list[tuple[float, int, int]] = []
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
        candidates.append((dist, centroid_x, centroid_y))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    _, cx_, cy_ = candidates[0]
    # 用最佳候选的 boundingRect 初始化 CSRT
    # 重新找这个候选的轮廓
    # 实际上我们已经有了 centroid，需要用 boundingRect
    # 重新从 contours 中找到对应的那个
    # 最简单的办法：用 centroid 反查
    for cnt in contours:
        if cv2.pointPolygonTest(cnt, (cx_, cy_), False) >= 0:
            _x, _y, _w, _h = cv2.boundingRect(cnt)
            return (_x, _y, _w, _h)

    return None


def refine_local_bbox(
    frame: np.ndarray,
    tx: int,
    ty: int,
    hsv_lower: tuple[int, int, int],
    hsv_upper: tuple[int, int, int],
    cx_global: int,
    cy_global: int,
    prev_w: int = 40,
    prev_h: int = 40,
    search_scale: float = 2.0,
    min_contour_area: int = 20,
) -> tuple[float, float, int, int]:
    """以 (tx, ty) 为中心，局部 HSV 掩模+轮廓检测，返回紧贴合的目标 bbox。

    Pipeline:
        1. 以 (tx, ty) 为中心截取 ROI（大小为 prev 尺寸 × search_scale）
        2. 在 ROI 上计算 HSV 二值化掩模
        3. 将全局准星 (cx_global, cy_global) 映射至 ROI 坐标系并涂黑
        4. 寻找最大轮廓 → boundingRect → 转换回全局坐标
        5. 无有效轮廓时退回 (tx, ty, prev_w, prev_h)

    Returns:
        ``(ball_x, ball_y, ball_w, ball_h)`` — 均为全局坐标。
    """
    h, w = frame.shape[:2]

    sw = max(int(prev_w * search_scale), 40)
    sh = max(int(prev_h * search_scale), 40)

    # ROI 边界（夹紧至画面内）
    x1 = max(0, tx - sw // 2)
    y1 = max(0, ty - sh // 2)
    x2 = min(w, x1 + sw)
    y2 = min(h, y1 + sh)
    if x2 - x1 < sw:
        x1 = max(0, x2 - sw)
    if y2 - y1 < sh:
        y1 = max(0, y2 - sh)

    roi = frame[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    local_mask = cv2.inRange(hsv_roi, hsv_lower, hsv_upper)

    # ---- 剔除局部 ROI 内的准星区域 ----
    rel_cx = cx_global - x1
    rel_cy = cy_global - y1
    if 0 <= rel_cx < (x2 - x1) and 0 <= rel_cy < (y2 - y1):
        cv2.circle(local_mask, (rel_cx, rel_cy), 10, 0, -1)

    contours, _ = cv2.findContours(
        local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= min_contour_area:
            rx, ry, rw, rh = cv2.boundingRect(largest)
            gx = x1 + rx
            gy = y1 + ry
            return (gx + rw / 2.0, gy + rh / 2.0, rw, rh)

    # Fallback：保持 CSRT 中心 + 上一帧尺寸
    return (float(tx), float(ty), prev_w, prev_h)


def sample_hsv_from_roi(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    h_margin: int = 10,
    s_margin: int = 50,
    v_margin: int = 50,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """从用户框选的 ROI 中采样中值 HSV，生成带容差的范围。

    特殊处理：若 ROI 中值 V（明度）< 50（纯黑/极暗目标），
    则强制返回泛用暗色阈值，忽略色相和饱和度干扰。

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

    # ---- 极暗目标：泛用暗色阈值 ----
    if v0 < 50:
        return (0, 0, 0), (180, 255, 60)

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
