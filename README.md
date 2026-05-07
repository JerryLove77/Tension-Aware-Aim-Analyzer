# Aim Tracking Analyzer

A tension-aware aim analyzer for FPS training videos. Tracks crosshair and target movement, computes kinematics metrics, and provides frame-by-frame VOD review.

## Features

- **Hybrid Tracking** — HSV color detection + CSRT tracker for robust target following
- **Kinematics Analysis** — Speed/acceleration mismatch, accuracy, PTC (Precision Tension Coefficient)
- **VOD Review** — Scrub through annotated frames with hit/miss visualization
- **Dark UI** — Clean PySide6 interface with real-time progress
- **Standalone EXE** — One-click build for distribution (no Python required for end users)

## Screenshot

```
┌─────────────────────────────────────────────┐
│  VIDEO FILE                                 │
│  [path/to/video.mp4              ] [BROWSE] │
│                                             │
│  START [0.0]  END [10.0]    [ ANALYZE ]     │
│                                             │
│  ANALYSIS PROGRESS                          │
│  [████████████████████░░░░░░░░░░] 65%       │
│                                             │
│  RESULTS                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │ Accuracy │ │ Avg Error│ │Speed Miss│     │
│  │  87.3%   │ │  4.2 px  │ │  0.234   │     │
│  └──────────┘ └──────────┘ └──────────┘     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │Accel Miss│ │   PTC    │ │Loss Count│     │
│  │  0.187   │ │  0.412   │ │    3     │     │
│  └──────────┘ └──────────┘ └──────────┘     │
│                                             │
│  VOD REVIEW                                 │
│  ┌─────────────────────────────────────┐    │
│  │                                     │    │
│  │        [annotated frame]            │    │
│  │                                     │    │
│  └─────────────────────────────────────┘    │
│  [━━━━━━━━━━━━━━━━░░░░░░░░░░░░░░░░░░]      │
└─────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- Windows 10/11

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/aim-tracking-analyzer.git
cd aim-tracking-analyzer

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-contrib-python` | Video capture, color detection, CSRT tracker |
| `PySide6` | GUI framework |
| `pandas` | Data processing |
| `scipy` | Signal smoothing (Savitzky-Golay), interpolation |
| `dxcam` | High-frequency screen capture (optional) |
| `tqdm` | Progress bars |

## Usage

### Run from Source

```bash
python main.py
```

### Build Standalone EXE

```bash
# One-click build
build.bat

# Or manually
python build_dist.py
```

Output: `dist/AimTrackingAnalyzer.exe` (~352 MB)

## How It Works

1. **Load Video** — Select an MP4/MKV/AVI file
2. **Set Time Range** — Choose start/end seconds for analysis
3. **Select Target** — Draw a box around the target in the OpenCV ROI window
4. **Analyze** — The app tracks the target across all frames
5. **Review** — Scrub through annotated frames in the VOD viewer

### Metrics Explained

| Metric | Description |
|--------|-------------|
| **Accuracy** | % of frames where crosshair overlaps target |
| **Avg Error** | Mean Euclidean distance (px) between crosshair and target center |
| **Speed Mismatch** | Normalized velocity difference between crosshair and target |
| **Accel Mismatch** | Normalized acceleration difference |
| **PTC** | Precision Tension Coefficient — accel mismatch during missed frames |
| **Loss Count** | Number of tracking-to-miss transitions |

## Project Structure

```
aim-tracking-analyzer/
├── main.py                 # Entry point
├── app/
│   ├── controller.py       # Business logic, thread management
│   └── views/
│       └── main_window.py  # PySide6 UI
├── vision/
│   ├── vision.py           # HSV detection, ROI sampling
│   ├── tracker.py          # Hybrid tracking engine
│   ├── video_reader.py     # Video processing utilities
│   ├── sampler.py          # Color sampling tools
│   └── capture.py          # Screen capture (dxcam)
├── analyzer/
│   ├── analysis.py         # Kinematics analysis (hardcore mode)
│   ├── smoothness.py       # Smoothness scoring
│   └── math_utils.py       # Signal smoothing, derivatives
├── tests/                  # Unit tests
├── build_dist.py           # PyInstaller build script
├── build.bat               # One-click build helper
├── utils.py                # Resource path helper
└── requirements.txt        # Python dependencies
```

## License

MIT

## Acknowledgments

- OpenCV for computer vision primitives
- PySide6 for the Qt-based GUI
- SciPy for signal processing
