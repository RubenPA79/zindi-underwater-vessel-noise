"""Build synthetic training mixtures from real backgrounds and vessel references.

A mixture is background energy plus a scaled reference energy. Because the gain
is chosen by us, ``share_added_vessel`` is known exactly, using the same
observed-cell rule as the leaderboard.
"""

from __future__ import annotations

import numpy as np

from src.audio import bell_envelope, passage_window, unit_energy_reference
from src.features import window_features
from src.layout import BAND_CENTRES, MINUTES, N_BANDS

DEFAULT_ZERO_FRACTION = 0.35
SHARE_MIN, SHARE_MAX = 1e-2, 0.79
MISSING_CELL_RATE = 0.033

# The test's missing cells are not random: they are whole minutes dropped at the
# edges of the window (620 of 1500 windows lack minute 1, 97 lack minute 15,
# eight lack two or three edge minutes). That is a recording cut on a minute
# grid, so the synthetic mixtures imitate it instead of dropping random cells.
FIRST_MINUTE_MISSING_RATE = 0.41
LAST_MINUTE_MISSING_RATE = 0.07
SECOND_MINUTE_MISSING_RATE = 0.03

_LOG_CENTRED = np.log10(BAND_CENTRES) - np.log10(BAND_CENTRES).mean()
_MAX_TILT_DB = 16.0


def build_references(
    levels_db: np.ndarray,
    rng: np.random.Generator,
    variants: int = 3,
) -> np.ndarray:
    """Turn per-vessel band levels into unit-energy ``(n, 15, 25)`` references.

    Each vessel gets several envelopes, from a narrow spike to an almost flat
    presence. The band profile also drifts across the window: a vessel passing a
    hydrophone changes its spectrum minute to minute, and that drift is exactly
    what breaks the coherence of a real background (median minute-to-minute
    spectral correlation 0.987 in the archive, 0.844 in the test).
    """
    references = []
    for levels in levels_db:
        for _ in range(variants):
            # From a one-minute spike (a 20 s CPA clip) to a nearly flat
            # presence: the challenge does not say how long the addition lasts.
            sigma = 10.0 ** rng.uniform(np.log10(0.3), np.log10(8.0))
            envelope = bell_envelope(sigma=sigma, centre=rng.uniform(3.0, 11.0))
            window = passage_window(levels, envelope)
            drift = np.linspace(-0.5, 0.5, MINUTES) * rng.uniform(-_MAX_TILT_DB, _MAX_TILT_DB)
            window = window + drift[:, None] * _LOG_CENTRED[None, :]
            references.append(unit_energy_reference(window))
    return np.stack(references)


def edge_minute_mask(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Observed mask where whole minutes drop out at the window edges.

    The test never has random missing cells: 620 of 1500 windows lack minute 1,
    97 lack minute 15 and eight lack two or three edge minutes. That is a
    recording cut on a minute grid, so the synthetic mixtures copy it.
    """
    observed = np.ones((n_samples, MINUTES, N_BANDS), dtype=bool)
    observed[rng.random(n_samples) < FIRST_MINUTE_MISSING_RATE, 0, :] = False
    observed[rng.random(n_samples) < LAST_MINUTE_MISSING_RATE, MINUTES - 1, :] = False
    second = rng.random(n_samples) < SECOND_MINUTE_MISSING_RATE
    if second.any():
        observed[second, 1, :] = False
    return observed


def observed_share(
    background_energy: np.ndarray,
    added_energy: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """``sum(added) / sum(total)`` over observed cells, matching the leaderboard."""
    added_observed = np.where(observed, added_energy, 0.0).sum(axis=(1, 2))
    total_observed = np.where(observed, background_energy + added_energy, 0.0).sum(axis=(1, 2))
    return np.divide(added_observed, total_observed, out=np.zeros_like(added_observed), where=total_observed > 0)


def sample_mixture_windows(
    backgrounds_db: np.ndarray,
    reference_energy: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    zero_fraction: float = DEFAULT_ZERO_FRACTION,
    share_range: tuple[float, float] = (SHARE_MIN, SHARE_MAX),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(mixture_db, target, background_index)`` for synthetic mixtures.

    ``share_range`` controls the prior on the added energy. A narrow range
    around 0.49 with ``zero_fraction=0.5`` reproduces a nearly binary target,
    which is what the public leaderboard suggests the real test looks like.
    """
    n_backgrounds = len(backgrounds_db)
    background_index = rng.integers(0, n_backgrounds, n_samples)
    reference_index = rng.integers(0, len(reference_energy), n_samples)

    background_energy = 10.0 ** (backgrounds_db[background_index].astype(np.float64) / 10.0)
    background_total = background_energy.sum(axis=(1, 2))

    share = np.zeros(n_samples)
    has_addition = rng.random(n_samples) >= zero_fraction
    low, high = share_range
    share[has_addition] = rng.uniform(low, high, has_addition.sum())

    # share = a / (b + a)  =>  a = share / (1 - share) * b
    added_total = share / (1.0 - share) * background_total
    added = added_total[:, None, None] * reference_energy[reference_index]
    mixture = background_energy + added

    observed = edge_minute_mask(n_samples, rng)
    target = observed_share(background_energy, added, observed)

    with np.errstate(divide="ignore"):
        mixture_db = np.where(observed, 10.0 * np.log10(mixture), np.nan)
    return mixture_db, target, background_index


def sample_mixtures(
    backgrounds_db: np.ndarray,
    reference_energy: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    zero_fraction: float = DEFAULT_ZERO_FRACTION,
    share_range: tuple[float, float] = (SHARE_MIN, SHARE_MAX),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(features, target, background_index)`` for synthetic mixtures."""
    mixture_db, target, background_index = sample_mixture_windows(
        backgrounds_db, reference_energy, n_samples, rng, zero_fraction, share_range
    )
    return window_features(mixture_db), target, background_index


def flat_reference(levels_db: np.ndarray) -> np.ndarray:
    """A reference present across the whole window with no envelope."""
    window = np.broadcast_to(levels_db, (MINUTES, N_BANDS)).copy()
    return unit_energy_reference(window)


def background_references(
    backgrounds_db: np.ndarray,
    site: np.ndarray,
    rng: np.random.Generator,
    per_site: int = 40,
    top_fraction: float = 0.05,
    floor_db: float = -30.0,
) -> np.ndarray:
    """Reference spectra taken from loud events in the archive itself.

    The HearMyShip recordings come from the Adriatic; the challenge mixes
    passages recorded in South African waters. For each site we take its
    loudest windows, subtract the site's median band spectrum in linear energy
    and keep the positive excess. That is a local, calibration-consistent stand
    in for a passing vessel.
    """
    energy = 10.0 ** (backgrounds_db.astype(np.float64) / 10.0)
    references: list[np.ndarray] = []

    for site_id in np.unique(site):
        site_energy = energy[site == site_id]
        median_energy = np.median(site_energy, axis=(0, 1))
        totals = site_energy.sum(axis=(1, 2))
        count = min(max(int(top_fraction * len(site_energy)), 5), per_site)
        for index in np.argsort(totals)[-count:]:
            excess = np.clip(site_energy[index].mean(axis=0) - median_energy, 0.0, None)
            if excess.sum() <= 0:
                continue
            levels = 10.0 * np.log10(excess + 1e-12)
            levels = np.maximum(levels, np.nanmax(levels) + floor_db)
            envelope = bell_envelope(sigma=rng.uniform(1.5, 8.0), centre=rng.uniform(4.0, 10.0))
            references.append(unit_energy_reference(passage_window(levels, envelope)))

    if not references:
        raise ValueError("no usable background references")
    return np.stack(references)
