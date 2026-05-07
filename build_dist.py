# filepath: build_dist.py
# ============================================================
# PyInstaller build script for Aim Tracking Analyzer
# Packages the app as a standalone Windows .exe
# ============================================================

import PyInstaller.__main__
import os
import sys

# Project root = this script's directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Entry point
ENTRY = os.path.join(PROJECT_ROOT, "main.py")

# Output directory
DIST_DIR = os.path.join(PROJECT_ROOT, "dist")
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")

# App name
APP_NAME = "AimTrackingAnalyzer"


def build() -> None:
    args = [
        ENTRY,
        f"--name={APP_NAME}",
        "--onefile",
        "--windowed",
        # Clean build artifacts
        "--clean",
        # Output directories
        f"--distpath={DIST_DIR}",
        f"--workpath={BUILD_DIR}",
        # Hidden imports that PyInstaller may miss
        "--hidden-import=cv2",
        "--hidden-import=cv2.cv2",
        "--hidden-import=numpy",
        "--hidden-import=pandas",
        "--hidden-import=scipy",
        "--hidden-import=scipy.signal",
        "--hidden-import=scipy.interpolate",
        "--hidden-import=PySide6",
        "--hidden-import=PySide6.QtWidgets",
        "--hidden-import=PySide6.QtCore",
        "--hidden-import=PySide6.QtGui",
        "--hidden-import=dxcam",
        "--hidden-import=tqdm",
        # Collect entire packages (OpenCV and PySide6 need this)
        "--collect-all=cv2",
        "--collect-all=PySide6",
        "--collect-all=scipy",
        # Exclude test modules
        "--exclude-module=tests",
        # Debug: uncomment next line if build fails
        # "--log-level=DEBUG",
    ]

    print("=" * 60)
    print(f"  Building {APP_NAME}.exe")
    print("=" * 60)
    print()

    PyInstaller.__main__.run(args)

    exe_path = os.path.join(DIST_DIR, f"{APP_NAME}.exe")
    if os.path.isfile(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print()
        print("=" * 60)
        print(f"  BUILD SUCCESSFUL")
        print(f"  Output: {exe_path}")
        print(f"  Size:   {size_mb:.1f} MB")
        print("=" * 60)
        print()
        print("To distribute to alpha testers:")
        print(f"  1. Share the file: {exe_path}")
        print("  2. Tell them to double-click to run")
        print("  3. No Python installation required!")
    else:
        print()
        print("BUILD FAILED — exe not found in dist/")
        sys.exit(1)


if __name__ == "__main__":
    build()
