# filepath: app/controller.py

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage, QPixmap

from vision import run_tracking_analysis, sample_hsv_from_roi
from analyzer.analysis import compute_metrics

_FALLBACK_HSV_LOWER = (0, 150, 150)
_FALLBACK_HSV_UPPER = (10, 255, 255)


class VideoAnalysisThread(QThread):
    """后台追踪 → 分析线程，不碰 UI 对象。"""

    progress_changed = Signal(int)
    results_ready = Signal(object)   # dict
    frame_data_ready = Signal(object)  # DataFrame (for VOD)
    finished = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        video_path: str,
        hsv_lower: tuple[int, int, int],
        hsv_upper: tuple[int, int, int],
        start_frame: int = 0,
        end_frame: int | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._video_path = video_path
        self._hsv_lower = hsv_lower
        self._hsv_upper = hsv_upper
        self._start_frame = start_frame
        self._end_frame = end_frame

    def run(self) -> None:
        try:
            def _on_progress(frac: float) -> None:
                self.progress_changed.emit(int(frac * 100))

            df = run_tracking_analysis(
                self._video_path,
                hsv_lower=self._hsv_lower,
                hsv_upper=self._hsv_upper,
                start_frame=self._start_frame,
                end_frame=self._end_frame,
                progress_callback=_on_progress,
            )

            self.progress_changed.emit(100)

            # Pass raw DataFrame for VOD before analysis
            self.frame_data_ready.emit(df)

            # Read FPS for analysis
            cap = cv2.VideoCapture(self._video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            results = compute_metrics(df, fps)
            self.results_ready.emit(results)

            n = len(df)
            self.finished.emit(
                f"Done — {n} frames  |  Acc: {results['accuracy']}%  |  "
                f"PTC: {results['ptc']}"
            )

        except Exception as exc:
            self.error.emit(str(exc))


class MainController:
    """协调 MainWindow ↔ tracking / analysis，含 VOD 渲染。"""

    def __init__(self, window) -> None:
        self._window = window
        self._thread: VideoAnalysisThread | None = None

        # VOD state
        self._vod_path: str | None = None
        self._vod_df: pd.DataFrame | None = None
        self._vod_cap: cv2.VideoCapture | None = None
        self._vod_fps: float = 0.0

        window.file_selected.connect(self._on_file_selected)
        window.analysis_requested.connect(self._on_analysis_requested)
        window.vod_slider.valueChanged.connect(self._on_slider_moved)

    # ==================================================================
    #  Slots
    # ==================================================================

    def _on_file_selected(self, path: str) -> None:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        cap.release()

        self._window.set_time_range(duration)
        self._window.set_info(f"Video loaded  ({duration:.1f}s  @ {fps:.0f} fps)")

    def _on_analysis_requested(self) -> None:
        self._close_vod()
        self._abort_thread()

        self._vod_path = self._window.file_path_edit.text()
        if not self._vod_path:
            self._window.set_info("No video selected")
            return

        # Read duration for clamping
        cap = cv2.VideoCapture(self._vod_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        self._vod_fps = fps

        start_time = self._window.get_start_time()
        end_time = self._window.get_end_time()
        start_frame = int(min(max(start_time, 0.0), duration) * fps)
        end_frame = int(min(max(end_time, start_time + 0.1), duration) * fps)
        cap.release()

        self._window.set_progress(0)
        self._window.set_info(
            f"Segment: {start_time:.1f}s–{end_time:.1f}s  "
            f"|  Select target color ..."
        )

        # Seek to start frame and show ROI
        cap2 = cv2.VideoCapture(self._vod_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame_at_start = cap2.read()
        cap2.release()

        if not ret or frame_at_start is None:
            self._window.set_info("Error: cannot seek to start frame")
            return

        roi = cv2.selectROI("Select Target Center", frame_at_start)
        cv2.destroyWindow("Select Target Center")

        if roi[2] == 0 or roi[3] == 0:
            hsv_lo, hsv_hi = _FALLBACK_HSV_LOWER, _FALLBACK_HSV_UPPER
        else:
            hsv_lo, hsv_hi = sample_hsv_from_roi(frame_at_start, roi)

        # Start thread
        self._thread = VideoAnalysisThread(
            self._vod_path, hsv_lo, hsv_hi,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        self._thread.progress_changed.connect(self._window.set_progress)
        self._thread.frame_data_ready.connect(self._on_frame_data)
        self._thread.results_ready.connect(self._on_results_ready)
        self._thread.finished.connect(self._window.set_info)
        self._thread.error.connect(
            lambda msg: self._window.set_info(f"Error: {msg}")
        )
        self._thread.start()

    def _on_frame_data(self, df: pd.DataFrame) -> None:
        """Store interpolated DataFrame and initialise VOD playback state."""
        # Interpolate NaN ball positions for smooth VOD rendering
        self._vod_df = df.copy()
        ball_cols = ["ball_x", "ball_y", "ball_w", "ball_h"]
        self._vod_df[ball_cols] = self._vod_df[ball_cols].interpolate(
            method="linear", limit_direction="both"
        )

        min_f, max_f = int(df["frame"].min()), int(df["frame"].max())
        self._window.set_vod_slider_range(min_f, max_f)

        # Open persistent capture for VOD scrubbing
        self._vod_cap = cv2.VideoCapture(self._vod_path)

        # Render first frame
        self._render_frame(min_f)

    def _on_results_ready(self, results: dict) -> None:
        self._window.populate_metrics(results)

    def _on_slider_moved(self, value: int) -> None:
        if self._vod_df is None:
            return
        self._render_frame(value)

    # ==================================================================
    #  VOD rendering
    # ==================================================================

    def _render_frame(self, frame_idx: int) -> None:
        """Read frame *frame_idx* from disk, draw annotations, push to UI."""
        if self._vod_cap is None or self._vod_df is None:
            return

        self._vod_cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = self._vod_cap.read()
        if not ret:
            return

        matches = self._vod_df[self._vod_df["frame"] == frame_idx]
        if matches.empty:
            return
        row = matches.iloc[0]

        bx, by = row["ball_x"], row["ball_y"]
        cx, cy = int(row["cross_x"]), int(row["cross_y"])

        # -- Dynamic line (green = on target, red = miss, AABB 1.0x) --
        if not pd.isna(bx) and not pd.isna(by):
            bxi, byi = int(bx), int(by)
            half_w = row["ball_w"] / 2.0
            half_h = row["ball_h"] / 2.0
            is_miss = (abs(bx - cx) > half_w) or (abs(by - cy) > half_h)
            line_color = (0, 0, 255) if is_miss else (0, 255, 0)  # BGR
            cv2.line(frame, (cx, cy), (bxi, byi), line_color, 2)

        # -- Crosshair (BGR orange 0,165,255) --
        cv2.circle(frame, (cx, cy), 8, (0, 165, 255), 2)
        cv2.line(frame, (cx - 14, cy), (cx + 14, cy), (0, 165, 255), 1)
        cv2.line(frame, (cx, cy - 14), (cx, cy + 14), (0, 165, 255), 1)

        # -- Target box (BGR green 0,255,0) --
        if not pd.isna(bx):
            bw, bh = int(row["ball_w"]), int(row["ball_h"])
            x0 = int(bx - bw / 2)
            y0 = int(by - bh / 2)
            cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), (0, 255, 0), 2)

        # Convert to QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self._window.set_vod_preview(pix)

    # ==================================================================
    #  Cleanup
    # ==================================================================

    def _close_vod(self) -> None:
        if self._vod_cap is not None:
            self._vod_cap.release()
            self._vod_cap = None
        self._vod_df = None

    def _abort_thread(self) -> None:
        if self._thread is not None:
            if self._thread.isRunning():
                self._thread.quit()
                self._thread.wait(2000)
            self._thread = None
