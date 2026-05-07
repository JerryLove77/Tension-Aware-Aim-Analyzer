# filepath: tests/test_video_reader.py

"""Standalone terminal harness to verify VideoProcessor.

Usage:
    python tests/test_video_reader.py <path-to-video>
"""

import sys

from vision.video_reader import VideoProcessor


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python tests/test_video_reader.py <video-file>")
        print("Falling back to dummy path 'test.mp4' for syntax check...")
        video_path = "test.mp4"
    else:
        video_path = sys.argv[1]

    try:
        with VideoProcessor(video_path) as proc:
            w, h = proc.frame_size
            print(f"File    : {proc.path}")
            print(f"Size    : {w} x {h}")
            print(f"Frames  : {proc.total_frames}")
            print(f"FPS     : {proc.fps:.2f}")
            print(f"Duration: {proc.duration_sec:.2f}s")
            print()

            count = 0
            for idx, frame in proc.extract_frames():
                count += 1
                if count <= 3:
                    print(f"  frame {idx:6d}  shape={frame.shape}")

            print(f"\nTotal frames read: {count}")

    except FileNotFoundError:
        print(f"[skip] '{video_path}' not found — video file needed for full test.")
        sys.exit(0)
    except RuntimeError as exc:
        print(f"[error] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
