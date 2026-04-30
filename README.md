# KovaaK Tracking Analyzer

A computer-vision based aim tracking analyzer for reviewing target tracking, spatial error, speed mismatch, and acceleration mismatch.

## Workflow

1. Calibration and tracking extraction:

```bash
streamlit run app.py
```

This exports:

- `output/calibration_raw.csv`
- `output/calib_config.json`

2. Physics analysis:

```bash
python Analyze.py --csv output/calibration_raw.csv --fps 360
```

This exports:

- `output/frame_errors.csv`
- `output/metrics.json`

3. Dashboard:

```bash
streamlit run dashboard.py
```

## Project Structure

```text
app.py                         Streamlit calibration/tracking UI
Analyze.py                     CLI entry for kinematics analysis
dashboard.py                   Streamlit dashboard UI
calibrate.py                   Optional OpenCV interactive calibration CLI
requirements.txt               Python dependencies
kovaak_tracker/
  analysis.py                  Kinematics extraction and metrics export
  calibration_cli.py           OpenCV mouse-pick calibration flow
  dashboard_data.py            Dashboard data loading, charts, and replay rendering
  settings.py                  Shared output paths
  tracking.py                  Tracking-by-detection pipeline
  video.py                     Video metadata, frame reading, upload temp files
  vision.py                    HSV sampling, color detection, tracker creation
```

The root scripts are intentionally thin. New features should usually be added inside `kovaak_tracker/`, then called from the matching UI or CLI entry.

## Install

```bash
pip install -r requirements.txt
```

For CSRT tracking, keep `opencv-contrib-python` installed. If only the base OpenCV package is available, the app falls back to KCF when possible.
