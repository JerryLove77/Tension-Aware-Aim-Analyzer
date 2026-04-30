import argparse

from kovaak_tracker.analysis import run_analysis


def print_report(metrics: dict) -> None:
    print("\n" + "=" * 70)
    print("  KovaaK Vectorized Physics Measurement Report")
    print("=" * 70)
    print(f"  Accuracy (On-Target)    : {metrics['accuracy']}%")
    print(f"  Speed Mismatch (dV)     : {metrics['speed_mismatch']} px/s")
    print(f"  Tension Mismatch (dA)   : {metrics['accel_mismatch']} px/s^2")
    print(f"  Pure Tension Coeff (PTC): {metrics['ptc']} Hz^2")
    print(f"  Global Avg Spatial Error: {metrics['avg_error_px']} px")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--fps", type=float, default=60)
    args = parser.parse_args()

    _, metrics, _ = run_analysis(args.csv, args.fps)
    print_report(metrics)


if __name__ == "__main__":
    main()
