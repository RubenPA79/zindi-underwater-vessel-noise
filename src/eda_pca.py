"""Diagnostic: do test windows leave the subspace where real backgrounds live?

A background window's 25 band levels live on a low-dimensional manifold (level,
slope, sea state). An added passage should push a window off that manifold. This
script fits PCA on backgrounds only and compares residuals for backgrounds and
test windows. If the test residuals are clearly larger, the residual is a usable
feature.

Run from ``proyectos/underwater-vessel-noise`` as::

    python -m src.eda_pca
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.layout import parse_columns, to_tensor

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
N_COMPONENTS = 8
QUANTILES = [10, 25, 50, 75, 90, 99]


def band_spectra(windows: np.ndarray) -> np.ndarray:
    return np.nanmean(windows, axis=1)


def report(name: str, values: np.ndarray) -> None:
    print(f"  {name:<12} " + "  ".join(f"{np.percentile(values, q):7.3f}" for q in QUANTILES))


def main() -> None:
    archive = np.load(DATA_DIR / "bg_windows.npz")
    backgrounds = archive["windows"]

    test = pd.read_csv(DATA_DIR / "Test.csv")
    features = list(test.columns[1:])
    minute_index, band_index = parse_columns(features)
    test_windows = to_tensor(test[features].to_numpy(dtype=float), minute_index, band_index)

    background_spectra = band_spectra(backgrounds)
    test_spectra = band_spectra(test_windows)

    scaler = StandardScaler().fit(background_spectra)
    pca = PCA(n_components=N_COMPONENTS, random_state=0).fit(scaler.transform(background_spectra))

    print(f"PCA on {len(background_spectra):,} background spectra, {N_COMPONENTS} components")
    print("explained variance ratio:", np.round(pca.explained_variance_ratio_, 3).tolist())
    print(f"cumulative: {pca.explained_variance_ratio_.sum():.3f}\n")

    def metrics(spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scaled = scaler.transform(spectra)
        scores = pca.transform(scaled)
        reconstructed = pca.inverse_transform(scores)
        residual = np.linalg.norm(scaled - reconstructed, axis=1)
        mahalanobis = np.sqrt((scores**2 / pca.explained_variance_).sum(axis=1))
        return residual, mahalanobis, scores[:, 0]

    background_residual, background_maha, _ = metrics(background_spectra)
    test_residual, test_maha, _ = metrics(test_spectra)

    print("percentiles:      " + "  ".join(f"{q:>7}%" for q in QUANTILES))
    print("residual norm")
    report("backgrounds", background_residual)
    report("test", test_residual)
    print("mahalanobis distance")
    report("backgrounds", background_maha)
    report("test", test_maha)

    separation = (np.median(test_residual) - np.median(background_residual)) / (
        np.std(background_residual) + 1e-9
    )
    print(f"\nseparation in background standard deviations: {separation:.2f}")
    print(f"test windows above background p90 residual: {100 * np.mean(test_residual > np.percentile(background_residual, 90)):.1f}%")


if __name__ == "__main__":
    main()
