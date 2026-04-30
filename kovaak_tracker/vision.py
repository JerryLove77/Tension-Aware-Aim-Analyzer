from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


Point = tuple[int, int]


def get_hsv_range(
    bgr_color: list[int] | tuple[int, int, int] | np.ndarray,
    is_crosshair: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an adaptive HSV range from a sampled BGR color."""
    hsv = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if v < 40:
        return np.array([0, 0, 0]), np.array([179, 255, min(255, v + 50)])

    if s < 30 and v > 200:
        return np.array([0, 0, max(0, v - 50)]), np.array([179, 50, 255])

    tolerance_h = 15 if is_crosshair else 10
    tolerance_sv = 60 if is_crosshair else 50
    return (
        np.array([max(0, h - tolerance_h), max(0, s - tolerance_sv), max(0, v - tolerance_sv)]),
        np.array([min(179, h + tolerance_h), min(255, s + tolerance_sv), min(255, v + tolerance_sv)]),
    )


def detect_point_by_color(
    frame: np.ndarray,
    hsv_lo: np.ndarray,
    hsv_hi: np.ndarray,
    *,
    min_area: float = 20,
    max_area_ratio: float = 0.05,
    max_aspect_ratio: float | None = None,
    ignore_top_ui: bool = False,
) -> tuple[Point | None, int | None, int | None]:
    """Find the color blob closest to the screen center."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    h_img, w_img = frame.shape[:2]
    center_x, center_y = w_img // 2, h_img // 2
    max_valid_area = w_img * h_img * max_area_ratio

    best_pos: Point | None = None
    best_w: int | None = None
    best_h: int | None = None
    min_dist = float("inf")

    for contour in contours:
        area = cv2.contourArea(contour)
        if not min_area < area < max_valid_area:
            continue

        _, _, width, height = cv2.boundingRect(contour)
        if height == 0:
            continue

        if max_aspect_ratio is not None and width / float(height) > max_aspect_ratio:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        if ignore_top_ui and cy < h_img * 0.12 and width > 60:
            continue

        dist = (cx - center_x) ** 2 + (cy - center_y) ** 2
        if dist < min_dist:
            min_dist = dist
            best_pos = (cx, cy)
            best_w = width
            best_h = height

    return best_pos, best_w, best_h


def detect_ball_by_color(
    frame: np.ndarray,
    hsv_lo: np.ndarray,
    hsv_hi: np.ndarray,
) -> tuple[Point | None, int | None, int | None]:
    """Detect the target ball with stricter filtering than a generic color blob."""
    return detect_point_by_color(
        frame,
        hsv_lo,
        hsv_hi,
        min_area=20,
        max_area_ratio=0.05,
        max_aspect_ratio=1.3,
        ignore_top_ui=True,
    )


def get_tracker(warn_callback: Callable[[str], None] | None = None):
    """Create the preferred OpenCV tracker, falling back when contrib APIs are absent."""
    try:
        return cv2.TrackerCSRT_create()
    except AttributeError:
        if warn_callback is not None:
            warn_callback("CSRT tracker not found; falling back to KCF. Install opencv-contrib-python for CSRT.")
        return cv2.TrackerKCF_create()


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def sample_median_bgr(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> list[int]:
    template = frame[int(y1) : int(y2), int(x1) : int(x2)]
    median_color = np.median(template.reshape(-1, 3), axis=0).astype(np.uint8)
    return median_color.tolist()
