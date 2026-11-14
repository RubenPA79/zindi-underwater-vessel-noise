"""Feature extraction for 15-minute x 25-band acoustic windows.

Everything works on decibel matrices and is NaN-aware: missing cells must stay
distinguishable from silence, so they are skipped rather than imputed.
"""

from __future__ import annotations

import warnings

import numpy as np

from src.layout import BAND_CENTRES, MINUTES

_LOG_FREQ = np.log10(BAND_CENTRES)
_LOG_FREQ_CENTRED = _LOG_FREQ - _LOG_FREQ.mean()
_LOG_FREQ_VARIANCE = float(np.sum(_LOG_FREQ_CENTRED**2))
N_FEATURES = 25 * 5 + 10


def _quiet_nan(fn, values: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(values, axis=axis)


def window_features(db: np.ndarray) -> np.ndarray:
    """Summary features for a batch of windows.

    Parameters
    ----------
    db:
        Array of shape ``(n_windows, 15, 25)`` holding level in dB, NaN where a
        cell is missing.

    Returns
    -------
    np.ndarray
        ``(n_windows, N_FEATURES)`` float32 matrix, NaN where a band is entirely
        missing in a window.
    """
    db = np.asarray(db, dtype=np.float64)
    if db.ndim != 3:
        raise ValueError(f"expected (n, 15, 25), got {db.shape}")

    mean = _quiet_nan(np.nanmean, db, axis=1)
    std = _quiet_nan(np.nanstd, db, axis=1)
    p10 = _nan_percentile(db, 10)
    p90 = _nan_percentile(db, 90)
    peak = _quiet_nan(np.nanmax, db, axis=1) - mean

    energy = 10.0 ** (db / 10.0)
    total_db = 10.0 * np.log10(_quiet_nan(np.nansum, energy, axis=(1, 2)))

    # Spectral slope: least-squares fit of band mean level against log frequency.
    deviation = mean - _quiet_nan(np.nanmean, mean, axis=1)[:, None]
    slope = np.nansum(_LOG_FREQ_CENTRED[None, :] * deviation, axis=1) / _LOG_FREQ_VARIANCE
    low_high = _quiet_nan(np.nanmean, mean[:, :5], axis=1) - _quiet_nan(np.nanmean, mean[:, -5:], axis=1)

    # Direct share estimates: a passage occupies part of the window, so the
    # quietest minute of each band is closer to the background than the mean is.
    total_energy = _quiet_nan(np.nansum, energy, axis=(1, 2))
    floor_energy = _quiet_nan(np.nanmin, energy, axis=1)
    low_energy = _nan_percentile(energy, 10)
    share_floor = 1.0 - _quiet_nan(np.nansum, floor_energy, axis=1) * MINUTES / total_energy
    share_low = 1.0 - _quiet_nan(np.nansum, low_energy, axis=1) * MINUTES / total_energy
    std_contrast = _quiet_nan(np.nanmean, std[:, :5], axis=1) - _quiet_nan(np.nanmean, std[:, -5:], axis=1)

    # Temporal coherence: a real background is one continuous recording, so its
    # band profile barely changes from minute to minute (median 0.987). A
    # passing vessel does change it, which breaks that coherence.
    coherence, coherence_std, coherence_last, slope_trend = _coherence_features(db)

    return np.column_stack(
        [
            mean,
            std,
            p10,
            p90,
            peak,
            total_db[:, None],
            slope[:, None],
            low_high[:, None],
            share_floor[:, None],
            share_low[:, None],
            std_contrast[:, None],
            coherence[:, None],
            coherence_std[:, None],
            coherence_last[:, None],
            slope_trend[:, None],
        ]
    ).astype(np.float32)


def _coherence_features(db: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Correlation between consecutive minutes' band profiles, plus a slope drift."""
    centred = db - _quiet_nan(np.nanmean, db, axis=2)[:, :, None]
    left = centred[:, :-1, :]
    right = centred[:, 1:, :]
    numerator = np.nansum(left * right, axis=2)
    denominator = np.sqrt(np.nansum(left * left, axis=2) * np.nansum(right * right, axis=2)) + 1e-30
    correlation = numerator / denominator

    coherence = _quiet_nan(np.nanmean, correlation, axis=1)
    coherence_std = _quiet_nan(np.nanstd, correlation, axis=1)
    coherence_last = correlation[:, -1]
    slope_trend = _spectral_slope(db[:, -5:, :]) - _spectral_slope(db[:, :5, :])
    return coherence, coherence_std, coherence_last, slope_trend


def _spectral_slope(db: np.ndarray) -> np.ndarray:
    mean = _quiet_nan(np.nanmean, db, axis=1)
    deviation = mean - _quiet_nan(np.nanmean, mean, axis=1)[:, None]
    return np.nansum(_LOG_FREQ_CENTRED[None, :] * deviation, axis=1) / _LOG_FREQ_VARIANCE


def _nan_percentile(db: np.ndarray, q: float) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(db, q, axis=1)
