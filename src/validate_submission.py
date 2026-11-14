#!/usr/bin/env python3
"""Validate a submission CSV before uploading it to Zindi.

Catches the failures that silently produce a 0 score: wrong IDs, wrong column
order, missing rows, NaN/out-of-range values and index columns left behind by
``to_csv``.

Example
-------
    python src/validate_submission.py submission.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = PROJECT_ROOT / "data" / "sample_submission.csv"
ID_COLUMN = "ID"
TARGET_COLUMN = "share_added_vessel"
EVIDENCE_COLUMN = "score_added_vessel"


def load_submission(path: Path, sample_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise SystemExit(f"error: submission not found: {path}")
    if not sample_path.exists():
        raise SystemExit(f"error: sample submission not found: {sample_path}")
    submission = pd.read_csv(path)
    sample = pd.read_csv(sample_path)
    return submission, sample


def check(submission: pd.DataFrame, sample: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    expected_columns = list(sample.columns)

    if list(submission.columns) != expected_columns:
        problems.append(f"columns are {list(submission.columns)}, expected {expected_columns}")
    if len(submission) != len(sample):
        problems.append(f"{len(submission)} rows, expected {len(sample)}")

    missing = set(sample[ID_COLUMN]) - set(submission[ID_COLUMN])
    extra = set(submission[ID_COLUMN]) - set(sample[ID_COLUMN])
    if missing:
        problems.append(f"{len(missing)} sample IDs missing (e.g. {sorted(missing)[:3]})")
    if extra:
        problems.append(f"{len(extra)} unknown IDs (e.g. {sorted(extra)[:3]})")
    duplicated = submission[ID_COLUMN][submission[ID_COLUMN].duplicated()]
    if not duplicated.empty:
        problems.append(f"{duplicated.nunique()} duplicated IDs (e.g. {duplicated.head(3).tolist()})")

    for column in (TARGET_COLUMN, EVIDENCE_COLUMN):
        if column not in submission.columns:
            continue
        values = pd.to_numeric(submission[column], errors="coerce")
        non_numeric = int(values.isna().sum() - submission[column].isna().sum())
        if non_numeric:
            problems.append(f"{column}: {non_numeric} non-numeric values")
        finite = values[np.isfinite(values)]
        if len(finite) != len(values):
            problems.append(f"{column}: {len(values) - len(finite)} NaN/inf values")
        if not finite.empty and (finite.min() < 0 or finite.max() > 1):
            problems.append(f"{column}: range [{finite.min():.4f}, {finite.max():.4f}] outside [0, 1]")

    return problems


def describe(submission: pd.DataFrame) -> None:
    for column in (TARGET_COLUMN, EVIDENCE_COLUMN):
        if column not in submission.columns:
            continue
        values = pd.to_numeric(submission[column], errors="coerce").dropna()
        if values.empty:
            continue
        quantiles = values.quantile([0.5, 0.9, 0.99]).round(4).to_dict()
        print(
            f"  {column}: mean={values.mean():.4f} std={values.std():.4f} "
            f"zeros={float((values == 0).mean()):.1%} median={quantiles[0.5]} "
            f"p90={quantiles[0.9]} p99={quantiles[0.99]}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    args = parser.parse_args(argv)

    submission, sample = load_submission(args.submission, args.sample)
    problems = check(submission, sample)

    print(f"rows={len(submission)} columns={list(submission.columns)}")
    describe(submission)

    if problems:
        print("\nINVALID:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nOK: submission matches sample_submission.csv and is ready to upload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
