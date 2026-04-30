import argparse

from kovaak_tracker.calibration_cli import run_calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    run_calibration(args.video, args.frames)


if __name__ == "__main__":
    main()
