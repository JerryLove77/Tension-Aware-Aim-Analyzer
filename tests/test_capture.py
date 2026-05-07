# filepath: tests/test_capture.py

"""Standalone terminal harness to verify ScreenCapturer FPS.

Usage:
    python tests/test_capture.py
"""

import time

from vision.capture import ScreenCapturer

DURATION_S = 10
TARGET_FPS = 60


def main() -> None:
    print(f"[harness] Starting capture test ({DURATION_S}s @ {TARGET_FPS} FPS target)")
    print(f"[harness] Press Ctrl+C to stop early\n")

    with ScreenCapturer(target_fps=TARGET_FPS) as capturer:
        start = time.monotonic()
        last_report = start
        frame_count = 0
        fps_samples: list[float] = []

        while True:
            now = time.monotonic()
            elapsed = now - start

            if elapsed >= DURATION_S:
                break

            frame = capturer.latest_frame()
            if frame is not None:
                frame_count += 1

            # Report every ~1 second
            if now - last_report >= 1.0:
                fps = frame_count / (now - last_report)
                fps_samples.append(fps)
                shape_info = (
                    f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "N/A"
                )
                print(
                    f"[{elapsed:6.2f}s]  "
                    f"Capture FPS: {fps:5.1f}  "
                    f"(frame: {shape_info})"
                )
                frame_count = 0
                last_report = now

            # Yield CPU — don't busy-wait on latest_frame
            time.sleep(0.001)

    avg_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0.0
    print(f"\n[harness] Done. Average FPS: {avg_fps:.1f} (target: {TARGET_FPS})")


if __name__ == "__main__":
    main()
