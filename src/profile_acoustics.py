"""Profile the South African acoustic parquet files.

Run from ``proyectos/underwater-vessel-noise`` as::

    python -m src.profile_acoustics
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.layout import BAND_CENTRES, match_band_columns

ACOUSTICS_DIR = Path(__file__).resolve().parents[1] / "data" / "acoustics"


def band_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if column.startswith("TOL_")]


def main() -> None:
    files = sorted(ACOUSTICS_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files in {ACOUSTICS_DIR}; run src.extract_archive first")

    total_rows = 0
    for path in files:
        frame = pd.read_parquet(path)
        total_rows += len(frame)
        tol = band_columns(list(frame.columns))
        matched, outside = match_band_columns(tol)
        ordered = [matched[band] for band in sorted(matched)]

        timestamps = pd.to_datetime(frame["t"], errors="coerce")
        duration = frame["dur"]
        print(f"\n{path.name}")
        print(f"  rows: {len(frame):,}   columns: {len(frame.columns)}   TOL columns: {len(tol)}")
        print(f"  time: {timestamps.min()} .. {timestamps.max()}")
        print(f"  dur (s): median {duration.median():.0f}, <55 s in {100 * (duration < 55).mean():.1f}% of rows")
        print(f"  sites: {frame['site'].nunique()}   recorders: {frame['recorder'].nunique()}")
        print(f"  common bands matched: {len(matched)} of 25   outside the grid: {len(outside)} {outside[:4]}")
        levels = frame[ordered].to_numpy(dtype=float)
        print(f"  missing TOL cells: {100 * (1 - np.isfinite(levels).mean()):.1f}%")
        print(f"  level range: {np.nanmin(levels):.1f} .. {np.nanmax(levels):.1f} dB")
        if frame["qc_25band_finite"].dtype == bool or set(frame["qc_25band_finite"].dropna().unique()) <= {0, 1}:
            print(f"  qc_25band_finite true: {100 * frame['qc_25band_finite'].mean():.1f}%")

    print(f"\nTOTAL rows across sites: {total_rows:,} (challenge states 1,715,779)")


if __name__ == "__main__":
    main()
