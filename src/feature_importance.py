"""Which features drive the prediction, grouped by family.

Trains one LightGBM on the synthetic mixtures and reports gain importance per
feature family. The point is interpretability for the write-up: which acoustic
evidence the model actually uses, and whether the temporal-coherence features
earn their place.

Run from ``proyectos/underwater-vessel-noise`` as::

    python -m src.feature_importance
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.features import window_features  # noqa: E402
from src.mixtures import sample_mixture_windows  # noqa: E402
from src.site_model import SiteMatcher  # noqa: E402
from src.train_baseline import PARAMS  # noqa: E402
from src.vtuad import vtuad_references  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROJECT_DIR = DATA_DIR.parent
FIGURES = PROJECT_DIR / "entrega" / "figures"
SEED = 2024


def feature_names(n_sites: int) -> tuple[list[str], list[str]]:
    """Individual feature names and the family each one belongs to."""
    names: list[str] = []
    families: list[str] = []

    for prefix, label in (
        ("band_mean", "Band mean level"),
        ("band_std", "Band temporal std"),
        ("band_p10", "Band 10th percentile"),
        ("band_p90", "Band 90th percentile"),
        ("band_peak", "Band peak minus mean"),
    ):
        for band in range(25):
            names.append(f"{prefix}_{band:02d}")
            families.append(label)

    singles = [
        ("total_level", "Total level", "Overall"),
        ("spectral_slope", "Spectral slope", "Overall"),
        ("low_high_ratio", "Low/high ratio", "Overall"),
        ("share_floor", "Share from temporal floor", "Direct share estimate"),
        ("share_low", "Share from 10th percentile", "Direct share estimate"),
        ("std_contrast", "Low/high std contrast", "Direct share estimate"),
        ("coherence_mean", "Coherence mean", "Temporal coherence"),
        ("coherence_std", "Coherence std", "Temporal coherence"),
        ("coherence_last", "Coherence last pair", "Temporal coherence"),
        ("slope_trend", "Spectral slope drift", "Temporal coherence"),
    ]
    for name, label, family in singles:
        names.append(name)
        families.append(family)

    for site in range(n_sites):
        names.append(f"site_excess_{site}")
        families.append("Site-matched excess")
    for name, family in (
        ("best_site_excess", "Site-matched excess"),
        ("best_site_id", "Site-matched excess"),
        ("share_from_best_site", "Site-matched excess"),
    ):
        names.append(name)
        families.append(family)

    return names, families


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    archive = np.load(DATA_DIR / "bg_windows.npz")
    backgrounds, site = archive["windows"], archive["site"]

    rng = np.random.default_rng(SEED)
    references = vtuad_references(
        ["pilotvessel", "rescue", "sailing", "tanker", "dredger", "pleasurecraft"], rng
    )
    mixtures, target, _ = sample_mixture_windows(
        backgrounds, references, 40_000, rng, 0.5, (0.44, 0.54)
    )

    matcher = SiteMatcher().fit(backgrounds, site)
    features = np.hstack([window_features(mixtures), matcher.transform(mixtures)])
    names, families = feature_names(len(matcher.sites_))
    if features.shape[1] != len(names):
        raise SystemExit(f"feature count mismatch: {features.shape[1]} vs {len(names)}")

    params = dict(PARAMS)
    params["learning_rate"] = 0.1
    model = lgb.train(params, lgb.Dataset(features, label=target), num_boost_round=300)

    importance = pd.DataFrame(
        {
            "feature": names,
            "family": families,
            "gain": model.feature_importance(importance_type="gain"),
        }
    )
    importance["share"] = importance["gain"] / importance["gain"].sum()

    by_family = (
        importance.groupby("family")["share"].sum().sort_values(ascending=False).reset_index()
    )
    importance.sort_values("share", ascending=False).to_csv(DATA_DIR / "feature_importance.csv", index=False)

    print("Importancia por familia (fraccion del gain total)")
    print(by_family.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nTop 10 variables individuales")
    print(importance.nlargest(10, "share")[["feature", "family", "share"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=160)

    axes[0].barh(by_family["family"][::-1], by_family["share"][::-1], color="#1b6ca8")
    axes[0].set_xlabel("Share of total gain")
    axes[0].set_title("Which feature families the model uses")
    axes[0].grid(axis="x", alpha=0.25, linestyle=":")

    top = importance.nlargest(12, "share")[::-1]
    axes[1].barh(top["feature"], top["share"], color="#c0392b")
    axes[1].set_xlabel("Share of total gain")
    axes[1].set_title("Top individual features")
    axes[1].tick_params(labelsize=8)
    axes[1].grid(axis="x", alpha=0.25, linestyle=":")

    figure.tight_layout()
    figure.savefig(FIGURES / "feature_importance.png")
    plt.close(figure)
    print(f"\nwrote {FIGURES.relative_to(PROJECT_DIR) / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
