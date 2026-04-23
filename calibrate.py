"""
kovaak_tracker/calibrate.py
============================
第一步：校准脚本 (V2 优化版 - 颜色特征提取)
- 从视频提取第一帧
- 用户点选小球颜色 (免疫背景干扰)
- 用户点选准星颜色
- 在前100帧跑检测，输出可视化验证视频

用法：
    python calibrate.py --video your_recording.mp4
"""

import cv2
import numpy as np
import argparse
import json
import os

calib = {
    "ball_bgr": None,
    "ball_hsv_lo": None,
    "ball_hsv_hi": None,
    "crosshair_bgr": None,
    "crosshair_hsv_lo": None,
    "crosshair_hsv_hi": None,
}

# ── 核心视觉算法 ──

def get_hsv_range(bgr_color, is_crosshair=False):
    """根据采样的 BGR 颜色，智能生成 HSV 容差范围"""
    hsv = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    
    if v < 40:  # 极暗/纯黑 (针对 Smoothbot)
        lo = np.array([0, 0, 0])
        hi = np.array([179, 255, v + 50])
    elif s < 30 and v > 200: # 极亮/纯白
        lo = np.array([0, 0, max(0, v - 50)])
        hi = np.array([179, 50, 255])
    else: # 常规颜色
        tolerance_h = 10 if not is_crosshair else 15
        tolerance_sv = 50 if not is_crosshair else 60
        lo = np.array([max(0, h-tolerance_h), max(0, s-tolerance_sv), max(0, v-tolerance_sv)])
        hi = np.array([min(179, h+tolerance_h), min(255, s+tolerance_sv), min(255, v+tolerance_sv)])
    
    return lo, hi

# 把 app.py 里的 detect_ball_by_color 替换为：
def detect_ball_by_color(frame, hsv_lo, hsv_hi):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None, None # 返回三个值
    
    h_img, w_img = frame.shape[:2]
    center_x, center_y = w_img // 2, h_img // 2
    max_valid_area = w_img * h_img * 0.05 
    
    best_pos, best_w, best_h = None, None, None
    min_dist = float('inf')
    
    for c in contours:
        area = cv2.contourArea(c)
        if 20 < area < max_valid_area:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist = (cx - center_x)**2 + (cy - center_y)**2
                
                if dist < min_dist:
                    min_dist = dist
                    best_pos = (cx, cy)
                    # 核心新增：提取这个轮廓的外接矩形宽高
                    _, _, w, h = cv2.boundingRect(c)
                    best_w, best_h = w, h
                    
    return best_pos, best_w, best_h

# ── 交互式颜色采样 ──

def mouse_pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr = param["frame"][y, x].tolist()
        param["picked"] = bgr
        print(f"  ✓ 采样成功 BGR={bgr}")

def select_color_interactive(frame, title_msg, is_crosshair=False):
    win = title_msg
    param = {"frame": frame.copy(), "picked": None}
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, mouse_pick_color, param)
    cv2.imshow(win, cv2.resize(frame, None, fx=1.5, fy=1.5))

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and param["picked"]: # Enter 确认
            cv2.destroyWindow(win)
            bgr = param["picked"]
            lo, hi = get_hsv_range(bgr, is_crosshair)
            return bgr, lo, hi
        elif key == 27: # Esc 取消
            cv2.destroyAllWindows()
            raise SystemExit("用户取消")

# ── 主流程 ──

def run_calibration(video_path, max_frames=100, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("无法读取视频第一帧")

    print("\n=== 第一步：点选【小球】颜色 ===")
    ball_bgr, b_lo, b_hi = select_color_interactive(first_frame, "【步骤1】点击小球采样颜色，按 Enter 确认", False)
    
    print("\n=== 第二步：点选【准星】颜色 ===")
    cross_bgr, c_lo, c_hi = select_color_interactive(first_frame, "【步骤2】点击准星采样颜色，按 Enter 确认", True)

    config = {
        "ball_bgr": ball_bgr,
        "ball_hsv_lo": b_lo.tolist(),
        "ball_hsv_hi": b_hi.tolist(),
        "crosshair_bgr": cross_bgr,
        "crosshair_hsv_lo": c_lo.tolist(),
        "crosshair_hsv_hi": c_hi.tolist(),
        "fps": fps,
        "resolution": [w, h],
    }
    
    with open(os.path.join(output_dir, "calib_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n=== 第三步：验证前 {max_frames} 帧检测效果 ===")
    out_path = os.path.join(output_dir, "calibration_check.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    results = []
    b_found, c_found = 0, 0

    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret: break

        vis = frame.copy()
        ball_pos, _, _ = detect_ball_by_color(frame, b_lo, b_hi)
        cross_pos, _, _ = detect_ball_by_color(frame, c_lo, c_hi)

        if ball_pos:
            b_found += 1
            cv2.circle(vis, ball_pos, 25, (0, 220, 80), 2)
            cv2.putText(vis, "Target", (ball_pos[0]+28, ball_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 80), 1)

        if cross_pos:
            c_found += 1
            cv2.drawMarker(vis, cross_pos, (0, 180, 255), cv2.MARKER_CROSS, 20, 2)

        if ball_pos and cross_pos:
            cv2.line(vis, ball_pos, cross_pos, (255, 200, 0), 1)
            dist = np.hypot(ball_pos[0]-cross_pos[0], ball_pos[1]-cross_pos[1])
            mid = ((ball_pos[0]+cross_pos[0])//2, (ball_pos[1]+cross_pos[1])//2)
            cv2.putText(vis, f"{dist:.0f}px", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        results.append({
            "frame": i, "time_s": round(i/fps, 3),
            "ball_x": ball_pos[0] if ball_pos else None,
            "ball_y": ball_pos[1] if ball_pos else None,
            "cross_x": cross_pos[0] if cross_pos else None,
            "cross_y": cross_pos[1] if cross_pos else None,
        })
        writer.write(vis)

    writer.release()
    cap.release()

    print(f"\n── 检测结果 ──")
    print(f"  小球识别率:  {b_found}/{max_frames} ({b_found/max_frames*100:.1f}%)")
    import pandas as pd
    pd.DataFrame(results).to_csv(os.path.join(output_dir, "calibration_raw.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    run_calibration(args.video, args.frames)