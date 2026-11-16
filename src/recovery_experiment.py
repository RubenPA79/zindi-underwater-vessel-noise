"""Can the added share be recovered without knowing the background?

The leaderboard tops out near 0.99, which implies somebody recovers the
controlled addition almost exactly. That is only possible if either the
background is known or the addition can be identified from the window alone.

This experiment measures the key half on *synthetic* mixtures, where the truth is
known: we are given the true clip, its offset and the true channel, and only have
to estimate the background from the window itself. That is the ceiling of any
background-free method. If it cannot recover the share, no amount of searching
over clips helps, and the 0.99 scores come from something we have not identified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BAND_CENTRES = 10.0 ** (1.5 + 0.1 * np.arange(25))


def to_energy(db: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(db), 10.0 ** (np.nan_to_num(db) / 10.0), 0.0)


def smooth_background(window_db: np.ndarray, support: np.ndarray, order: int = 1) -> np.ndarray:
    """Fit a low-order polynomial in time per band, using minutes outside the support."""
    minutes = np.arange(window_db.shape[0], dtype=float)
    outside = ~support
    if outside.sum() <= order + 1:
        outside = np.ones_like(support, dtype=bool)
    design = np.vander(minutes, order + 1, increasing=True)
    coefficients, *_ = np.linalg.lstsq(design[outside], window_db[outside], rcond=None)
    return design @ coefficients


def make_mixture(rng, backgrounds: np.ndarray, levels: np.ndarray, channel_db: float = 3.0):
    """Background + (channel-tilted) clip at a random offset, in linear energy."""
    background_db = backgrounds[rng.integers(len(backgrounds))]
    clip_db = levels[rng.integers(len(levels))]

    valid = np.isfinite(clip_db).any(axis=1)
    clip = clip_db[valid]
    if len(clip) == 0:
        return None
    length = min(len(clip), background_db.shape[0])
    clip = clip[:length]

    offset = int(rng.integers(0, background_db.shape[0] - length + 1))
    log_f = np.log10(BAND_CENTRES)
    log_f = (log_f - log_f.mean()) / log_f.std()
    tilt = rng.uniform(-channel_db, channel_db) * log_f + rng.uniform(-channel_db, channel_db) * (log_f**2 - 1.0)

    background = to_energy(background_db)
    addition = to_energy(clip + tilt[None, :]) * rng.uniform(0.5, 2.0)
    if addition.sum() <= 0:
        return None
    addition *= (rng.uniform(0.44, 0.54) * background.sum()) / addition.sum()

    mixture = background.copy()
    support = np.zeros(background.shape[0], dtype=bool)
    mixture[offset : offset + length] += addition
    support[offset : offset + length] = True

    placed = np.zeros_like(background)
    placed[offset : offset + length] = addition

    observed = np.isfinite(background_db)
    return mixture, placed, support, observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixtures", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    backgrounds = np.load(DATA_DIR / "bg_windows.npz")["windows"]
    levels = np.load(DATA_DIR / "vtuad_strict_levels.npz")["levels"]
    print(f"backgrounds: {backgrounds.shape}  clips: {levels.shape}", flush=True)

    rng = np.random.default_rng(args.seed)
    errors: list[float] = []

    while len(errors) < args.mixtures:
        made = make_mixture(rng, backgrounds, levels)
        if made is None:
            continue
        mixture, addition, support, observed = made

        true_share = addition[observed].sum() / mixture[observed].sum()
        mixture_db = 10.0 * np.log10(np.where(mixture > 0, mixture, np.nan))

        background_hat = smooth_background(mixture_db, support)
        residual = np.where(observed, mixture - to_energy(background_hat), 0.0)
        estimated_share = np.clip(residual, 0.0, None)[observed].sum() / mixture[observed].sum()

        errors.append(estimated_share - true_share)

    errors = np.array(errors)
    print(f"RMSE: {np.sqrt((errors ** 2).mean()):.4f}")
    print(f"sesgo: {errors.mean():+.4f}  |  p90 |error|: {np.percentile(np.abs(errors), 90):.4f}")
    print(f"fraccion con |error| < 0.05: {(np.abs(errors) < 0.05).mean():.1%}")


if __name__ == "__main__":
    main()
