"""Build contiguous 15-minute background windows from the acoustic parquet files.

Each site is a one-minute time series. A window is 15 consecutive minutes in
which all 25 common bands are present and within a plausible level range.
Windows never overlap.

Writes ``data/bg_windows.npz`` with ``windows`` (float32, n x 15 x 25, dB),
``site`` (int16) and ``start`` (datetime64[s]).

Run from ``proyectos/underwater-vessel-noise`` as::

    python -m src.build_windows
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.layout import BAND_CENTRES, MINUTES, ordered_band_columns, parse_columns, to_tensor

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ACOUSTICS_DIR = DATA_DIR / "acoustics"
OUTPUT = DATA_DIR / "bg_windows.npz"

STEP_SECONDS = 60
STEP_TOLERANCE_SECONDS = 5.0
# Test levels run 29..153 dB; the archive also carries sentinel rows far below 0.
MIN_PLAUSIBLE_DB = 10.0
MAX_PLAUSIBLE_DB = 170.0
MAX_WINDOWS_PER_SITE = 6000
RANDOM_SEED = 0


def non_overlapping_starts(valid: np.ndarray, step_ok: np.ndarray, window: int) -> np.ndarray:
    """Start indices of non-overlapping runs of ``window`` valid consecutive rows."""
    starts: list[int] = []
    run = 0
    for i in range(len(valid)):
        if not valid[i]:
            run = 0
            continue
        run = 1 if (i == 0 or not step_ok[i]) else run + 1
        if run == window:
            starts.append(i - window + 1)
            run = 0
    return np.asarray(starts, dtype=np.int64)


def site_windows(path: Path, rng: np.random.Generator) -> tuple[np.ndarray, int, np.ndarray]:
    """Return usable windows for one site, its numeric id and their start times."""
    band_columns = ordered_band_columns(pq.ParquetFile(path).schema.names)
    frame = pd.read_parquet(path, columns=["t", "site"] + band_columns)
    frame = frame.sort_values("t").reset_index(drop=True)

    levels = frame[band_columns].to_numpy(dtype=np.float32)
    valid = np.isfinite(levels).all(axis=1)
    valid &= (levels >= MIN_PLAUSIBLE_DB).all(axis=1) & (levels <= MAX_PLAUSIBLE_DB).all(axis=1)

    seconds = frame["t"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    step_ok = np.zeros(len(seconds), dtype=bool)
    step_ok[1:] = np.abs(np.diff(seconds) - STEP_SECONDS) <= STEP_TOLERANCE_SECONDS

    starts = non_overlapping_starts(valid, step_ok, MINUTES)
    if len(starts) > MAX_WINDOWS_PER_SITE:
        starts = np.sort(rng.choice(starts, size=MAX_WINDOWS_PER_SITE, replace=False))

    indices = starts[:, None] + np.arange(MINUTES)
    windows = levels[indices]
    site_id = int(str(frame["site"].iloc[0])[:2])
    return windows, site_id, seconds[starts].astype("datetime64[s]")


def window_mean_db(windows: np.ndarray) -> np.ndarray:
    """Mean level of each window, averaged in linear energy then back to dB."""
    energy = 10.0 ** (windows.astype(np.float64) / 10.0)
    mean_energy = np.nanmean(energy, axis=(1, 2))
    return 10.0 * np.log10(mean_energy)


def load_test_windows() -> np.ndarray:
    test = pd.read_csv(DATA_DIR / "Test.csv")
    features = list(test.columns[1:])
    minute_index, band_index = parse_columns(features)
    return to_tensor(test[features].to_numpy(dtype=float), minute_index, band_index)


def compare_with_test(windows: np.ndarray, test: np.ndarray) -> None:
    background = window_mean_db(windows)
    tested = window_mean_db(test)
    quantiles = [1, 5, 25, 50, 75, 95, 99]

    print("\nMean window level (dB), background vs test")
    print(f"  {'q':>4} {'background':>12} {'test':>10}")
    for q in quantiles:
        print(f"  {q:>3}% {np.percentile(background, q):>12.1f} {np.nanpercentile(tested, q):>10.1f}")

    background_band = np.nanmean(windows, axis=(0, 1))
    test_band = np.nanmean(test, axis=(0, 1))
    print("\nPer-band mean level (dB), low and high end")
    for band in (0, 4, 9, 14, 19, 24):
        print(
            f"  {BAND_CENTRES[band]:>7.1f} Hz  background {background_band[band]:>6.1f} "
            f" test {test_band[band]:>6.1f}"
        )

    print(f"\nTest windows louder than the background median: {100 * np.mean(tested > np.median(background)):.1f}%")


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    files = sorted(ACOUSTICS_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files in {ACOUSTICS_DIR}")

    all_windows, all_sites, all_starts = [], [], []
    for path in files:
        windows, site_id, starts = site_windows(path, rng)
        all_windows.append(windows)
        all_sites.append(np.full(len(windows), site_id, dtype=np.int16))
        all_starts.append(starts)
        print(
            f"{path.stem:<22} windows: {len(windows):>6,}  "
            f"mean level {window_mean_db(windows).mean():>6.1f} dB"
        )

    windows = np.concatenate(all_windows)
    sites = np.concatenate(all_sites)
    starts = np.concatenate(all_starts)
    print(f"\ntotal background windows: {len(windows):,} from {len(np.unique(sites))} sites")

    np.savez_compressed(OUTPUT, windows=windows, site=sites, start=starts)
    print(f"saved {OUTPUT.relative_to(DATA_DIR.parent)} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")

    compare_with_test(windows, load_test_windows())


if __name__ == "__main__":
    main()
