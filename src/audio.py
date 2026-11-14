"""Convert vessel recordings into the challenge's 15-minute x 25-band layout.

Absolute calibration of the recordings is unknown and irrelevant here: the
reference is only ever added with a chosen energy gain, so what matters is the
band *shape* and the temporal envelope, not the absolute level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

from src.layout import BAND_CENTRES, MINUTES, N_BANDS

# Each band spans half a deci-decade either side of its centre.
_BAND_LOW = BAND_CENTRES / 10**0.05
_BAND_HIGH = BAND_CENTRES * 10**0.05
_FLOOR = 1e-30


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    """Read a WAV file as mono float64 in roughly [-1, 1]."""
    sample_rate, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float64)
    return int(sample_rate), data


def band_levels(signal: np.ndarray, sample_rate: int, nperseg: int = 16384) -> np.ndarray:
    """Third-octave band levels (dB, arbitrary reference) of a recording."""
    nperseg = min(nperseg, len(signal))
    frequencies, psd = welch(signal, fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2)

    levels = np.empty(N_BANDS, dtype=np.float64)
    for index, (low, high) in enumerate(zip(_BAND_LOW, _BAND_HIGH)):
        inside = (frequencies >= low) & (frequencies < high)
        if inside.sum() >= 2:
            power = np.trapz(psd[inside], frequencies[inside])
        else:
            power = psd[int(np.argmin(np.abs(frequencies - BAND_CENTRES[index])))]
        levels[index] = 10.0 * np.log10(power + _FLOOR)
    return levels


def bell_envelope(minutes: int = MINUTES, sigma: float = 3.0, centre: float = 7.0) -> np.ndarray:
    """Smooth passage envelope: it enters, peaks and leaves within the window."""
    positions = np.arange(minutes, dtype=np.float64)
    envelope = np.exp(-0.5 * ((positions - centre) / sigma) ** 2)
    return envelope / envelope.max()


def passage_window(
    levels: np.ndarray,
    envelope: np.ndarray | None = None,
) -> np.ndarray:
    """Expand band levels into a ``(15, 25)`` dB matrix following the envelope."""
    envelope = bell_envelope() if envelope is None else np.asarray(envelope, dtype=np.float64)
    safe = np.maximum(envelope, 1e-6)
    # Energy scaling per minute: (amplitude ratio)**2 in dB.
    return levels[None, :] + 20.0 * np.log10(safe)[:, None]


def unit_energy_reference(window_db: np.ndarray) -> np.ndarray:
    """Normalise a reference window so its total linear energy sums to one."""
    energy = 10.0 ** (window_db / 10.0)
    return energy / energy.sum()
