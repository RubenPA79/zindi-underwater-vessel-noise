"""Acoustic window layout and the official target definition.

The test set stores one row per 15-minute window with 375 columns named
``m<minute>_TOL_<centre_frequency>``. Levels are decibels, so they cannot be
added: everything must go through linear energy (``10 ** (dB / 10)``).

The leaderboard target is

    share_added_vessel = sum(added energy) / sum(total energy)

both sums taken over the same *observed* minute-band cells, which is what
``synthetic_share`` implements.
"""

from __future__ import annotations

import re

import numpy as np

MINUTES = 15
N_BANDS = 25
# Third-octave-ish bands on a base-10 deci-decade grid: 31.6 Hz .. 7.94 kHz.
BAND_CENTRES = np.array([31.6 * 10 ** (k / 10) for k in range(N_BANDS)])

_COLUMN_RE = re.compile(r"^m(\d{1,2})_TOL_(\d+(?:\.\d+)?)$")


class LayoutError(ValueError):
    """Raised when the feature columns do not follow the expected schema."""


def parse_columns(columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Map feature column names to their (minute, band) index.

    Returns two integer arrays of shape ``(n_features,)`` holding the 0-based
    minute and band index for each column, in the order given.
    """
    minutes: list[int] = []
    bands: list[int] = []
    for column in columns:
        match = _COLUMN_RE.match(column)
        if not match:
            raise LayoutError(f"column does not match 'mNN_TOL_<freq>': {column!r}")
        minute = int(match.group(1)) - 1
        if not 0 <= minute < MINUTES:
            raise LayoutError(f"minute index out of range in {column!r}")
        frequency = float(match.group(2))
        # Match to the nearest nominal centre so '7943' and '7943.0' both work.
        band = int(np.argmin(np.abs(BAND_CENTRES - frequency)))
        if abs(BAND_CENTRES[band] - frequency) > BAND_CENTRES[band] * 0.02:
            raise LayoutError(f"frequency {frequency} is not a known band centre")
        minutes.append(minute)
        bands.append(band)

    minute_index = np.asarray(minutes, dtype=int)
    band_index = np.asarray(bands, dtype=int)
    if len(set(map(tuple, zip(minute_index, band_index)))) != len(columns):
        raise LayoutError("duplicated minute-band cells in the feature columns")
    return minute_index, band_index


def to_tensor(values: np.ndarray, minute_index: np.ndarray, band_index: np.ndarray) -> np.ndarray:
    """Reshape ``(n_windows, n_features)`` levels into ``(n_windows, 15, 25)``.

    Unobserved cells stay NaN, which keeps "missing" distinct from "silence".
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise LayoutError(f"expected a 2-D array, got shape {values.shape}")
    n_windows = values.shape[0]
    tensor = np.full((n_windows, MINUTES, N_BANDS), np.nan, dtype=float)
    tensor[:, minute_index, band_index] = values
    return tensor


def db_to_energy(level_db: np.ndarray | float) -> np.ndarray:
    """Convert a level in dB to linear energy. NaN is preserved."""
    return 10.0 ** (np.asarray(level_db, dtype=float) / 10.0)


def energy_to_db(energy: np.ndarray | float) -> np.ndarray:
    """Convert linear energy back to dB, keeping NaN."""
    energy = np.asarray(energy, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(energy)


def synthetic_share(
    background_db: np.ndarray,
    added_energy: np.ndarray | float,
    valid: np.ndarray | None = None,
) -> float:
    """Compute ``share_added_vessel`` for a background plus a known addition.

    ``background_db`` is a level in dB. ``added_energy`` is the passage
    reference in **linear energy**, not dB: a level of 0 dB is an energy of 1,
    so "no addition" can only be expressed as an energy of 0. Convert dB
    levels with :func:`db_to_energy` first.

    Both arguments are broadcastable to ``background_db``. Only cells where the
    background is finite and ``valid`` is true take part in both sums, matching
    the official definition.
    """
    background = np.asarray(background_db, dtype=float)
    observed = np.isfinite(background)
    if valid is not None:
        observed &= np.asarray(valid, dtype=bool)

    added = np.broadcast_to(np.asarray(added_energy, dtype=float), background.shape)
    background_energy = db_to_energy(background)[observed]
    added_energy_observed = added[observed]
    if background_energy.size == 0:
        raise LayoutError("no observed cells to score")

    total = background_energy + added_energy_observed
    return float(added_energy_observed.sum() / total.sum())


def assert_full_grid(minute_index: np.ndarray, band_index: np.ndarray) -> None:
    """Check that the columns cover every minute-band cell exactly once."""
    cells = np.zeros((MINUTES, N_BANDS), dtype=int)
    np.add.at(cells, (minute_index, band_index), 1)
    if not np.all(cells == 1):
        missing = int((cells == 0).sum())
        duplicated = int((cells > 1).sum())
        raise LayoutError(f"incomplete grid: {missing} cells missing, {duplicated} duplicated")


_TOL_RE = re.compile(r"^TOL_(\d+(?:\.\d+)?)$")


def match_band_columns(columns: list[str]) -> tuple[dict[int, str], list[str]]:
    """Map ``TOL_<freq>`` columns onto the 25-band grid.

    Sites export extra bands below the common range (down to 0.8 Hz), so
    unmatched names are returned rather than raising. The first column wins
    when two names resolve to the same band.
    """
    matched: dict[int, str] = {}
    outside: list[str] = []
    for column in columns:
        match = _TOL_RE.match(column)
        if not match:
            continue
        frequency = float(match.group(1))
        band = int(np.argmin(np.abs(BAND_CENTRES - frequency)))
        if abs(BAND_CENTRES[band] - frequency) > BAND_CENTRES[band] * 0.02 or band in matched:
            outside.append(column)
        else:
            matched[band] = column
    return matched, outside


def ordered_band_columns(columns: list[str]) -> list[str]:
    """The 25 common-band columns in ascending frequency order."""
    matched, _ = match_band_columns(columns)
    if len(matched) != N_BANDS:
        raise LayoutError(f"found {len(matched)} of {N_BANDS} common bands")
    return [matched[band] for band in range(N_BANDS)]
