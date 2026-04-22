🎯 Tension-Aware Aim-Analyzer
A physics-based aim tracking analyzer designed to diagnose Muscle Tension and Speed Matching using computer vision.

🌊 Core Concept: Tension Vector Analysis
This tool implements a unique J/E (Jitter/Error) Ratio model to identify the root cause of tracking errors:

Jitter (J): The relative acceleration of your crosshair.

Error (E): The spatial distance between your crosshair and the target.

Tension Balance Ratio (TBR): J/E

High TBR (> 1.8): Indicates Over-tension (Grip is too tight, leading to excessive jitter).

Low TBR (< 0.6): Indicates Under-tension (Reaction is lagging, hand is too relaxed).

🚀 Workflow
Calibration (app.py):

Bash
streamlit run app.py
Trim your VOD to the relevant tracking segment.

Sample the target color to initialize the CV tracker.

Physics Analysis (Analyze.py):

Bash
python Analyze.py --csv output/calibration_raw.csv --fps 360
Processes frame data and generates the Tension Diagnosis.

Visualization Dashboard (dashboard.py):

Bash
streamlit run dashboard.py
View your Tension Quadrant and interactive VOD replay with dynamic hitboxes.

🛠️ Requirements
Python 3.10+

OpenCV, Streamlit, Pandas, Plotly, SciPy

📝 Acknowledgments
Inspired by MattyOW's theories on tension management and speed matching.

Developed by Jianrui (Jerry) Zhang.
