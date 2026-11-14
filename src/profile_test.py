"""Profile Test.csv, the sample submission and the acoustic archive.

Run from ``proyectos/underwater-vessel-noise`` as::

    python -m src.profile_test
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.layout import BAND_CENTRES, MINUTES, N_BANDS, assert_full_grid, parse_columns, to_tensor

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ARCHIVE = "South_African_Real_Acoustics.zip"


def profile_test() -> None:
    test = pd.read_csv(DATA_DIR / "Test.csv")
    print(f"Test.csv: {test.shape[0]} rows x {test.shape[1]} columns")

    id_column = test.columns[0]
    print(f"  first column: {id_column!r}; example IDs: {test[id_column].head(3).tolist()}")
    print(f"  duplicated IDs: {int(test[id_column].duplicated().sum())}")

    features = list(test.columns[1:])
    print(f"  feature columns: {len(features)} (expected {MINUTES * N_BANDS})")
    print(f"  first three: {features[:3]}")
    print(f"  last three:  {features[-3:]}")

    minute_index, band_index = parse_columns(features)
    assert_full_grid(minute_index, band_index)
    print("  column schema: OK, full 15x25 grid, each cell exactly once")

    values = test[features].to_numpy(dtype=float)
    tensor = to_tensor(values, minute_index, band_index)
    finite = np.isfinite(tensor)
    print(f"  missing cells: {100 * (1 - finite.mean()):.1f}% ({int((~finite).sum())} of {tensor.size})")
    print(f"  level range: {np.nanmin(tensor):.1f} .. {np.nanmax(tensor):.1f} dB")
    print(f"  level quantiles: {np.round(np.nanpercentile(tensor, [1, 5, 50, 95, 99]), 1).tolist()}")

    rows_complete = finite.all(axis=(1, 2)).mean()
    print(f"  windows with no missing cell: {100 * rows_complete:.1f}%")
    per_window_missing = (1 - finite.mean(axis=(1, 2)))
    print(f"  missing per window: median {np.median(per_window_missing):.1%}, max {per_window_missing.max():.1%}")

    band_mean = np.nanmean(tensor, axis=(0, 1))
    print("  loudest bands:", [f"{BAND_CENTRES[i]:.0f} Hz ({band_mean[i]:.1f} dB)" for i in np.argsort(band_mean)[-3:][::-1]])

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    print(f"\nsample_submission.csv: {sample.shape[0]} rows, columns {list(sample.columns)}")
    same_ids = set(sample.iloc[:, 0]) == set(test[id_column])
    print(f"  IDs match Test.csv: {same_ids}")
    print(f"  placeholder values: {sample.iloc[:3, 1:].to_numpy().tolist()}")


def profile_archive() -> None:
    archive_path = DATA_DIR / ARCHIVE
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
    print(f"\n{ARCHIVE}: {len(members)} entries")
    for name in members[:15]:
        print(f"  {name}")
    if len(members) > 15:
        print(f"  ... and {len(members) - 15} more")


def main() -> None:
    profile_test()
    profile_archive()


if __name__ == "__main__":
    main()
