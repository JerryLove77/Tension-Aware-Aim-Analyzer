# filepath: utils.py
# ============================================================
# Resource path helper for PyInstaller compatibility
# ============================================================

import os
import sys


def resource_path(relative_path: str) -> str:
    """Get absolute path to a resource, works for dev and PyInstaller.

    When running as a PyInstaller bundle, resources are extracted to a
    temporary folder referenced by sys._MEIPASS.  In normal dev mode,
    the resource is relative to the script's own directory.
    """
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)
