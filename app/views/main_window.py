# filepath: app/views/main_window.py

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QLineEdit, QDoubleSpinBox,
    QProgressBar, QFrame, QSlider, QFileDialog,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap


DARK_STYLESHEET = """
QMainWindow {
    background-color: #0d0d0d;
}
QLabel#sectionLabel {
    color: #aaaaaa;
    font-size: 10px;
    font-family: "Consolas", "Courier New", monospace;
    letter-spacing: 2px;
    padding-bottom: 2px;
}
QLineEdit#filePath {
    color: #cccccc;
    background-color: #1a1a1a;
    border: 1px solid #333333;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
    font-family: "Consolas", "Courier New", monospace;
}
QDoubleSpinBox {
    color: #cccccc;
    background-color: #1a1a1a;
    border: 1px solid #333333;
    border-radius: 4px;
    padding: 4px 6px;
    font-size: 12px;
    font-family: "Consolas", "Courier New", monospace;
}
QDoubleSpinBox:focus {
    border-color: #00ccff;
}
QPushButton#btnBrowse {
    color: #ffaa00;
    background-color: #1f180d;
    border: 1px solid #ffaa00;
    font-size: 12px;
    font-family: "Consolas", monospace;
    font-weight: bold;
    padding: 6px 18px;
    border-radius: 6px;
}
QPushButton#btnBrowse:hover {
    background-color: #ffaa00;
    color: #0d0d0d;
}
QPushButton#btnAnalyze {
    color: #00ff88;
    background-color: #0d1f12;
    border: 1px solid #00ff88;
    font-size: 13px;
    font-family: "Consolas", monospace;
    font-weight: bold;
    padding: 6px 24px;
    border-radius: 6px;
}
QPushButton#btnAnalyze:hover {
    background-color: #00ff88;
    color: #0d0d0d;
}
QPushButton#btnAnalyze:disabled {
    color: #335544;
    border-color: #335544;
    background-color: #0d1f12;
}
QProgressBar {
    background-color: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    text-align: center;
    font-size: 12px;
    font-family: "Consolas", monospace;
    color: #cccccc;
    height: 20px;
}
QProgressBar::chunk {
    background-color: #00ccff;
    border-radius: 5px;
}
QLabel#infoLabel {
    color: #888888;
    font-size: 12px;
    font-family: "Consolas", monospace;
    padding: 2px 0;
}
/* Metric cards */
QFrame#metricCard {
    background-color: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 6px;
}
QLabel#metricTitle {
    color: #777777;
    font-size: 9px;
    font-family: "Consolas", monospace;
}
QLabel#metricValue {
    font-size: 22px;
    font-family: "Consolas", monospace;
    font-weight: bold;
}
/* VOD */
QLabel#vodPreview {
    background-color: #0a0a0a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
}
QSlider::groove:horizontal {
    border: 1px solid #333333;
    height: 6px;
    background: #1a1a1a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #00ccff;
    border: none;
    width: 14px;
    height: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #00ccff;
    border-radius: 3px;
}
"""


class MainWindow(QMainWindow):
    file_selected = Signal(str)
    analysis_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Tension-Aware Aim Analyzer")
        self.resize(760, 820)
        self.setStyleSheet(DARK_STYLESHEET)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(10)

        # ============================================================
        #  1.  VIDEO FILE
        # ============================================================
        self._add_section_label(layout, "VIDEO FILE")

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setObjectName("filePath")
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("No file selected...")

        self.btn_browse = QPushButton("BROWSE")
        self.btn_browse.setObjectName("btnBrowse")

        file_row = QHBoxLayout()
        file_row.setSpacing(10)
        file_row.addWidget(self.file_path_edit, stretch=1)
        file_row.addWidget(self.btn_browse)
        layout.addLayout(file_row)

        # ============================================================
        #  2.  TIME RANGE + ANALYZE
        # ============================================================
        time_row = QHBoxLayout()
        time_row.setSpacing(8)

        self._add_label_widget(time_row, "START")
        self.spin_start = QDoubleSpinBox()
        self.spin_start.setDecimals(1)
        self.spin_start.setRange(0.0, 999.0)
        self.spin_start.setSingleStep(0.5)
        self.spin_start.setValue(0.0)
        time_row.addWidget(self.spin_start)
        time_row.addSpacing(4)

        self._add_label_widget(time_row, "END")
        self.spin_end = QDoubleSpinBox()
        self.spin_end.setDecimals(1)
        self.spin_end.setRange(0.0, 999.0)
        self.spin_end.setSingleStep(0.5)
        self.spin_end.setValue(10.0)
        time_row.addWidget(self.spin_end)
        time_row.addSpacing(12)

        self.btn_analyze = QPushButton("ANALYZE")
        self.btn_analyze.setObjectName("btnAnalyze")
        self.btn_analyze.setEnabled(False)
        time_row.addWidget(self.btn_analyze)
        time_row.addStretch()

        layout.addLayout(time_row)

        # ============================================================
        #  3.  PROGRESS
        # ============================================================
        self._add_section_label(layout, "ANALYSIS PROGRESS")
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setValue(0)

        self.info_label = QLabel("Select a video file and click ANALYZE")
        self.info_label.setObjectName("infoLabel")

        layout.addWidget(self.progress_bar)
        layout.addWidget(self.info_label)

        # ============================================================
        #  4.  METRICS CARDS  (3 × 2 grid)
        # ============================================================
        self._add_section_label(layout, "RESULTS")

        self._metrics: dict[str, QLabel] = {}
        grid = QGridLayout()
        grid.setSpacing(8)

        card_defs = [
            ("Accuracy",       "accuracy",          "#00ff88"),
            ("Avg Error",      "avg_error_px",      "#ffaa00"),
            ("Speed Mismatch", "avg_speed_mismatch", "#00ccff"),
            ("Accel Mismatch", "avg_accel_mismatch", "#cc88ff"),
            ("PTC",            "ptc",               "#ff6b6b"),
            ("Loss Count",     "loss_count",        "#ff6b6b"),
        ]

        for i, (title, key, color) in enumerate(card_defs):
            card, val_label = self._build_metric_card(title, color)
            self._metrics[key] = val_label
            grid.addWidget(card, i // 3, i % 3)

        layout.addLayout(grid)

        # ============================================================
        #  5.  VOD REVIEW
        # ============================================================
        self._add_section_label(layout, "VOD REVIEW")

        self.vod_label = QLabel()
        self.vod_label.setObjectName("vodPreview")
        self.vod_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vod_label.setMinimumHeight(240)
        self.vod_label.setText("—")
        layout.addWidget(self.vod_label, stretch=1)

        self.vod_slider = QSlider(Qt.Orientation.Horizontal)
        self.vod_slider.setEnabled(False)
        self.vod_slider.setMinimum(0)
        self.vod_slider.setMaximum(0)
        layout.addWidget(self.vod_slider)

        # ------------------------------------------------------------
        #  Connections
        # ------------------------------------------------------------
        self.btn_browse.clicked.connect(self._on_browse)
        self.btn_analyze.clicked.connect(self._on_analyze_clicked)

    # ==================================================================
    #  Internal helpers
    # ==================================================================

    @staticmethod
    def _add_section_label(parent: QVBoxLayout, text: str) -> None:
        lbl = QLabel(text)
        lbl.setObjectName("sectionLabel")
        parent.addWidget(lbl)

    @staticmethod
    def _add_label_widget(parent: QHBoxLayout, text: str) -> None:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #888888; font-size: 11px; font-family: Consolas, monospace;")
        parent.addWidget(lbl)

    def _build_metric_card(self, title: str, accent: str) -> tuple[QFrame, QLabel]:
        frame = QFrame()
        frame.setObjectName("metricCard")

        vbox = QVBoxLayout(frame)
        vbox.setContentsMargins(10, 6, 10, 6)
        vbox.setSpacing(2)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("metricTitle")

        val_lbl = QLabel("—")
        val_lbl.setObjectName("metricValue")
        val_lbl.setStyleSheet(f"color: {accent};")

        vbox.addWidget(title_lbl)
        vbox.addWidget(val_lbl)

        return frame, val_lbl

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a video file",
            "",
            "Video Files (*.mp4 *.mkv *.avi *.mov *.flv *.wmv);;All Files (*)",
        )
        if not path:
            return
        self.file_path_edit.setText(path)
        self.btn_analyze.setEnabled(True)
        self.info_label.setText(f"Ready — {path.split('/')[-1].split('\\')[-1]}")
        self.file_selected.emit(path)

    def _on_analyze_clicked(self) -> None:
        self.progress_bar.setValue(0)
        self._clear_metrics()
        self.vod_label.clear()
        self.vod_label.setText("—")
        self.vod_slider.setEnabled(False)
        self.analysis_requested.emit()

    def _clear_metrics(self) -> None:
        for lbl in self._metrics.values():
            lbl.setText("—")

    # ==================================================================
    #  Public API  (called from controller)
    # ==================================================================

    def set_progress(self, percent: int) -> None:
        self.progress_bar.setValue(max(0, min(100, percent)))

    def set_info(self, text: str) -> None:
        self.info_label.setText(text)

    def populate_metrics(self, results: dict) -> None:
        mapping = {
            "accuracy":          lambda v: f"{v}%",
            "avg_error_px":      lambda v: f"{v} px",
            "loss_count":        lambda v: str(v),
            "avg_speed_mismatch": lambda v: f"{v:.3f}",
            "avg_accel_mismatch": lambda v: f"{v:.3f}",
            "ptc":               lambda v: f"{v:.3f}",
        }
        for key, fmt in mapping.items():
            if key in self._metrics and key in results:
                self._metrics[key].setText(fmt(results[key]))

    def set_time_range(self, max_time: float) -> None:
        self.spin_start.setMaximum(max_time)
        self.spin_end.setMaximum(max_time)
        self.spin_end.setValue(max_time)

    def get_start_time(self) -> float:
        return self.spin_start.value()

    def get_end_time(self) -> float:
        return self.spin_end.value()

    def set_vod_preview(self, pixmap: QPixmap) -> None:
        scaled = pixmap.scaled(
            self.vod_label.width() or 640,
            self.vod_label.height() or 360,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.vod_label.setPixmap(scaled)

    def set_vod_slider_range(self, fmin: int, fmax: int) -> None:
        self.vod_slider.setMinimum(fmin)
        self.vod_slider.setMaximum(fmax)
        self.vod_slider.setValue(fmin)
        self.vod_slider.setEnabled(True)
