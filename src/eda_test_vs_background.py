"""Compare test windows against real background windows on a few fixed statistics.

The test is built from backgrounds plus a controlled addition, so any statistic
that separates the two populations is a candidate detector. Results are printed
as deciles and a coarse histogram, which is enough to see whether the test
response is bimodal.

Run from ``proyectos/underwater-vessel-noise`` as::

    python -m src.eda_test_vs_background
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.layout import BAND_CENTRES, parse_columns, to_tensor

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def total_level_db(windows: np.ndarray) -> np.ndarray:
    energy = 10.0 ** (windows.astype(np.float64) / 10.0)
    return 10.0 * np.log10(np.nansum(energy, axis=(1, 2)))


def share_floor(windows: np.ndarray) -> np.ndarray:
    energy = 10.0 ** (windows.astype(np.float64) / 10.0)
    total = np.nansum(energy, axis=(1, 2))
    floor = np.nanmin(energy, axis=1)
    return 1.0 - np.nansum(floor, axis=1) * windows.shape[1] / total


def band_excess(windows: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Mean per-band z-score against the background corpus, in dB space."""
    corpus_mean = np.nanmean(reference, axis=(0, 1))
    corpus_std = np.nanstd(reference, axis=(0, 1)) + 1e-9
    band_mean = np.nanmean(windows, axis=1)
    return np.nanmean((band_mean - corpus_mean) / corpus_std, axis=1)


def summarise(name: str, values: np.ndarray) -> None:
    quantiles = np.percentile(values, [1, 10, 25, 50, 75, 90, 99])
    print(f"{name:<22} " + "  ".join(f"{q:6.2f}" for q in quantiles))


def histogram(name: str, values: np.ndarray, low: float, high: float, bins: int = 12) -> None:
    counts, edges = np.histogram(values, bins=bins, range=(low, high))
    peak = max(counts.max(), 1)
    print(f"\n{name} ({low}..{high})")
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(40 * count / peak)
        print(f"  {left:7.2f}..{right:7.2f} {count:5d} {bar}")


def main() -> None:
    archive = np.load(DATA_DIR / "bg_windows.npz")
    backgrounds = archive["windows"]

    test = pd.read_csv(DATA_DIR / "Test.csv")
    features = list(test.columns[1:])
    minute_index, band_index = parse_columns(features)
    test_windows = to_tensor(test[features].to_numpy(dtype=float), minute_index, band_index)

    print(f"backgrounds: {backgrounds.shape[0]:,}   test: {test_windows.shape[0]:,}")
    print("\npercentiles                 1%     10%     25%     50%     75%     90%     99%")
    summaries = {
        "total level (dB)": (total_level_db(backgrounds), total_level_db(test_windows)),
        "share_floor": (share_floor(backgrounds), share_floor(test_windows)),
        "band excess (z)": (band_excess(backgrounds, backgrounds), band_excess(test_windows, backgrounds)),
    }
    for name, (background_values, test_values) in summaries.items():
        summarise(f"{name} [bg]", background_values)
        summarise(f"{name} [test]", test_values)
        print()

    low, high = np.percentile(np.concatenate(list(summaries["band excess (z)"])), [1, 99])
    histogram("band excess, backgrounds", summaries["band excess (z)"][0], low, high)
    histogram("band excess, test", summaries["band excess (z)"][1], low, high)

    print("\nloudest test bands vs corpus (dB):")
    corpus_band = np.nanmean(backgrounds, axis=(0, 1))
    test_band = np.nanmean(test_windows, axis=(0, 1))
    for band in range(0, 25, 4):
        print(f"  {BAND_CENTRES[band]:>7.1f} Hz  corpus {corpus_band[band]:6.1f}  test {test_band[band]:6.1f}")


if __name__ == "__main__":
    main()
