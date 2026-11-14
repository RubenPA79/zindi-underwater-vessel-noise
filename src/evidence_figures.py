"""Evidence figures for the research write-up.

Three panels that carry the argument:

1. Minute-to-minute spectral coherence: real backgrounds are one continuous
   recording, so their band profile barely changes; the test breaks that.
2. Mean band profile of the archive versus the test.
3. PCA residual of the band spectrum against the background subspace: the test
   does NOT leave it, which is why shape-only features saturate.

Run from ``proyectos/underwater-vessel-noise`` as::

    python -m src.evidence_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.layout import BAND_CENTRES, parse_columns, to_tensor  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROJECT_DIR = DATA_DIR.parent
FIGURES = PROJECT_DIR / "entrega" / "figures"
N_COMPONENTS = 8


def load_test() -> np.ndarray:
    test = pd.read_csv(DATA_DIR / "Test.csv")
    features = list(test.columns[1:])
    minute_index, band_index = parse_columns(features)
    return to_tensor(test[features].to_numpy(dtype=float), minute_index, band_index)


def coherence(windows: np.ndarray) -> np.ndarray:
    centred = windows - np.nanmean(windows, axis=2, keepdims=True)
    left, right = centred[:, :-1, :], centred[:, 1:, :]
    numerator = np.nansum(left * right, axis=2)
    denominator = np.sqrt(np.nansum(left * left, axis=2) * np.nansum(right * right, axis=2)) + 1e-30
    return np.nanmean(numerator / denominator, axis=1)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    backgrounds = np.load(DATA_DIR / "bg_windows.npz")["windows"]
    test = load_test()

    background_coherence = coherence(backgrounds)
    test_coherence = coherence(test)

    background_band = np.nanmean(backgrounds, axis=(0, 1))
    test_band = np.nanmean(test, axis=(0, 1))

    background_spectra = np.nanmean(backgrounds, axis=1)
    test_spectra = np.nanmean(test, axis=1)
    scaler = StandardScaler().fit(background_spectra)
    pca = PCA(n_components=N_COMPONENTS, random_state=0).fit(scaler.transform(background_spectra))

    def residual(spectra: np.ndarray) -> np.ndarray:
        scaled = scaler.transform(spectra)
        rebuilt = pca.inverse_transform(pca.transform(scaled))
        return np.linalg.norm(scaled - rebuilt, axis=1)

    background_residual = residual(background_spectra)
    test_residual = residual(test_spectra)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=160)

    bins = np.linspace(0, 1, 41)
    axes[0].hist(background_coherence, bins=bins, alpha=0.6, label="Archive backgrounds", color="#1b6ca8", density=True)
    axes[0].hist(test_coherence, bins=bins, alpha=0.6, label="Test windows", color="#c0392b", density=True)
    axes[0].axvline(np.median(background_coherence), color="#1b6ca8", linestyle="--", linewidth=1.4)
    axes[0].axvline(np.median(test_coherence), color="#c0392b", linestyle="--", linewidth=1.4)
    axes[0].set_xlabel("Minute-to-minute spectral coherence")
    axes[0].set_ylabel("Density")
    axes[0].set_title(
        f"Coherence: archive {np.median(background_coherence):.2f} vs test {np.median(test_coherence):.2f}"
    )
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(alpha=0.25, linestyle=":")

    axes[1].plot(BAND_CENTRES, background_band, marker="o", markersize=3, label="Archive backgrounds", color="#1b6ca8")
    axes[1].plot(BAND_CENTRES, test_band, marker="s", markersize=3, label="Test windows", color="#c0392b")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Band centre frequency (Hz)")
    axes[1].set_ylabel("Mean band level (dB, export convention)")
    axes[1].set_title("Band profile: the test is louder across all bands")
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].grid(alpha=0.25, linestyle=":")

    axes[2].hist(background_residual, bins=40, alpha=0.6, label="Archive backgrounds", color="#1b6ca8", density=True)
    axes[2].hist(test_residual, bins=40, alpha=0.6, label="Test windows", color="#c0392b", density=True)
    axes[2].set_xlabel(f"Residual outside the {N_COMPONENTS}-component background subspace")
    axes[2].set_ylabel("Density")
    axes[2].set_title(
        f"Shape residual: archive {np.median(background_residual):.3f} vs test {np.median(test_residual):.3f}"
    )
    axes[2].legend(fontsize=8, frameon=False)
    axes[2].grid(alpha=0.25, linestyle=":")

    figure.suptitle("Acoustic evidence: the addition changes level and coherence, not spectral shape", fontsize=11)
    figure.tight_layout()
    output = FIGURES / "evidence.png"
    figure.savefig(output)
    plt.close(figure)

    print(f"wrote {output.relative_to(PROJECT_DIR)}")
    print(f"coherence median: backgrounds {np.median(background_coherence):.4f}  test {np.median(test_coherence):.4f}")
    print(f"test windows below 0.8 coherence: {100 * np.mean(test_coherence < 0.8):.1f}% "
          f"(backgrounds {100 * np.mean(background_coherence < 0.8):.1f}%)")
    print(f"residual median: backgrounds {np.median(background_residual):.4f}  test {np.median(test_residual):.4f}")
    print(f"mean level gap: {np.mean(test_band - background_band):.2f} dB")


if __name__ == "__main__":
    main()
