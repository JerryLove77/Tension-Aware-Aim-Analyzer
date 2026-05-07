# Build Instructions - Aim Tracking Analyzer

## Quick Build (Recommended)

Double-click `build.bat` in the project root. It will:
1. Activate the virtual environment
2. Install PyInstaller if needed
3. Run the build script
4. Place the executable in `dist/`

## Manual Build

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install PyInstaller (if not already installed)
pip install pyinstaller>=6.0.0

# Run build script
python build_dist.py
```

## Output

The build produces a single standalone executable:

```
dist/AimTrackingAnalyzer.exe  (~352 MB)
```

## Distribution

Share `AimTrackingAnalyzer.exe` with alpha testers. They can:
1. Double-click to run
2. No Python installation required
3. No additional files needed

## Files Created

| File | Purpose |
|------|---------|
| `build_dist.py` | PyInstaller build script |
| `build.bat` | One-click build helper |
| `utils.py` | Resource path helper for PyInstaller compatibility |
| `requirements-build.txt` | Build dependencies (PyInstaller) |
| `AimTrackingAnalyzer.spec` | Auto-generated PyInstaller spec |

## Notes

- The executable is large (~352 MB) because it bundles PySide6, OpenCV, SciPy, and all dependencies
- First launch may take a few seconds while the OS loads the bundled libraries
- The app uses `cv2.selectROI` which opens a native OpenCV window for target selection
- All analysis runs locally — no internet connection required
