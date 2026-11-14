"""Recalibrate a submission toward a target mean and a chosen amount of model signal.

The synthetic prior used for training does not have to match the real target
distribution. When a reliable statistic about the true target is available (for
example the organisers' constant benchmark), the predictions can be shifted to
that mean and shrunk so that model variance is only kept in proportion to the
confidence we have in it.

    calibrated = mean + alpha * (prediction - prediction.mean())

``alpha = 0`` reproduces the constant benchmark; ``alpha = 1`` keeps the full
model spread but recentres it.

Run from ``proyectos/underwater-vessel-noise`` as::

    python -m src.recalibrate --input submission_final.csv --mean 0.2461 --alpha 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
TARGET_COLUMN = "share_added_vessel"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--mean", type=float, required=True, help="target mean of share_added_vessel")
    parser.add_argument("--alpha", type=float, default=0.5, help="how much of the model spread to keep")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output or args.input.with_name(args.input.stem + "_calibrated.csv")

    frame = pd.read_csv(args.input)
    prediction = frame[TARGET_COLUMN]
    calibrated = (args.mean + args.alpha * (prediction - prediction.mean())).clip(0.0, 1.0)

    frame[TARGET_COLUMN] = calibrated
    frame["score_added_vessel"] = calibrated
    frame.to_csv(output, index=False)

    print(f"input  mean {prediction.mean():.4f}  std {prediction.std():.4f}")
    print(f"output mean {calibrated.mean():.4f}  std {calibrated.std():.4f}  alpha {args.alpha}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
