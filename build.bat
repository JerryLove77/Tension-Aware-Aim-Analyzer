@echo off
echo ============================================================
echo   Aim Tracking Analyzer - Build Script
echo ============================================================
echo.

REM Check if we're in the right directory
if not exist "main.py" (
    echo ERROR: main.py not found. Run this script from the project root.
    pause
    exit /b 1
)

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if pyinstaller is installed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    pip install pyinstaller>=6.0.0
)

echo.
echo Building executable...
echo.
python build_dist.py

echo.
echo Build process complete!
pause
