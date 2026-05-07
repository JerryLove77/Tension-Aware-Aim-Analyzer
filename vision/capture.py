# filepath: vision/capture.py

from __future__ import annotations

from types import TracebackType
from typing import Self

import dxcam
import numpy as np


class ScreenCapturer:
    """High-frequency screen capture wrapper around dxcam.

    This module is part of the ``vision`` layer and MUST NOT import
    anything from PySide6 or the ``app`` package.
    """

    def __init__(
        self,
        target_fps: int = 60,
        region: tuple[int, int, int, int] | None = None,
        monitor: int = 0,
    ) -> None:
        self._target_fps = target_fps
        self._region = region
        self._monitor = monitor
        self._camera: dxcam.DXCamera | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._camera is not None and self._camera.is_capturing

    def start(self) -> None:
        """Begin background capture at the configured FPS."""
        if self.is_running:
            return
        self._camera = dxcam.create(
            output_idx=self._monitor,
            output_color="BGR",
        )
        self._camera.start(target_fps=self._target_fps, region=self._region)

    def stop(self) -> None:
        """Stop capture and release the DXGI resource."""
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
            self._camera = None

    def latest_frame(self) -> np.ndarray | None:
        """Return the most recently captured frame, non-blocking."""
        if self._camera is None:
            return None
        return self._camera.get_latest_frame()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop()
