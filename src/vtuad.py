"""VTUAD vessel passages converted exactly as the challenge starter notebook does.

The notebook's ``minute_band_power`` is the reference conversion: non-overlapping
2-second segments, Hann window, one-sided PSD, integrate over each deci-decade
band, then average 30 segments per minute in linear units. Reproducing it keeps
our references in the same representation the organisers used.

VTUAD is the corpus named in the notebook ("Adapted vessel-passage components
derive from VTUAD ... v1.3.2", CC BY 4.0, doi:10.5281/zenodo.23037895).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import periodogram

from src.layout import BAND_CENTRES, MINUTES, N_BANDS

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
VTUAD_DIR = DATA_DIR / "vtuad"
_BAND_LOW = BAND_CENTRES / 10**0.05
_BAND_HIGH = BAND_CENTRES * 10**0.05


def minute_band_power(samples: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """One-channel audio -> complete-minute band power, shape ``(minutes, 25)``."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("Use one channel with finite samples.")
    sample_rate = float(sample_rate_hz)
    segment_size = int(2 * sample_rate)
    minutes = samples.size // (30 * segment_size)
    if minutes == 0:
        raise ValueError("At least one complete minute is needed.")

    segments = samples[: minutes * 30 * segment_size].reshape(-1, segment_size)
    frequencies, psd = periodogram(
        segments,
        fs=sample_rate,
        window="hann",
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=1,
    )

    power = np.full((len(segments), N_BANDS), np.nan)
    bandwidth = frequencies[1] - frequencies[0]
    for band, (low, high) in enumerate(zip(_BAND_LOW, _BAND_HIGH)):
        bins = (frequencies >= low) & (frequencies < high)
        if high <= sample_rate / 2 and bins.any():
            power[:, band] = psd[:, bins].sum(axis=1) * bandwidth
    return power.reshape(minutes, 30, N_BANDS).mean(axis=1)


def extract_class(class_name: str) -> list[Path]:
    """Unpack a per-class archive (if needed) and return its WAV paths."""
    target = VTUAD_DIR / class_name
    if not target.exists():
        archive = VTUAD_DIR / f"vtuad_wav_{class_name}.zip"
        if not archive.exists():
            raise FileNotFoundError(f"missing archive {archive}")
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(target)
    return sorted(target.rglob("*.wav"))


def clip_levels(path: Path) -> np.ndarray:
    """Band levels in dB for one clip, shape ``(minutes, 25)``."""
    sample_rate, samples = wavfile.read(path)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    power = minute_band_power(samples.astype(np.float64), sample_rate)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(power + 1e-30)


def collect_levels(class_name: str) -> tuple[np.ndarray, list[str]]:
    """All clips of a class as a padded ``(n_clips, 15, 25)`` dB tensor.

    Clips shorter than the window are padded with silence (NaN), which the
    mixer treats as "no added energy at that minute".
    """
    levels_list: list[np.ndarray] = []
    names: list[str] = []
    for path in extract_class(class_name):
        levels = clip_levels(path)
        window = np.full((MINUTES, N_BANDS), np.nan)
        take = min(len(levels), MINUTES)
        window[:take] = levels[:take]
        levels_list.append(window)
        names.append(path.name)
    if not levels_list:
        raise ValueError(f"no clips found for class {class_name}")
    return np.stack(levels_list), names


def place_clip(levels_db: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Drop a clip into a 15-minute window at a random position.

    The challenge varied "time trajectories ... before combination with measured
    South African backgrounds", so the passage can start anywhere.
    """
    valid = np.isfinite(levels_db).any(axis=1)
    minutes = levels_db[valid]
    if len(minutes) > MINUTES:
        start = int(rng.integers(0, len(minutes) - MINUTES + 1))
        minutes = minutes[start : start + MINUTES]

    window = np.full((MINUTES, N_BANDS), np.nan)
    offset = int(rng.integers(0, MINUTES - len(minutes) + 1))
    window[offset : offset + len(minutes)] = minutes
    return window


def to_unit_energy(window_db: np.ndarray) -> np.ndarray:
    """dB window (NaN = absent) to a unit-energy ``(15, 25)`` matrix."""
    energy = np.where(np.isfinite(window_db), 10.0 ** (np.nan_to_num(window_db) / 10.0), 0.0)
    total = energy.sum()
    if total <= 0:
        raise ValueError("empty reference window")
    return energy / total


def vtuad_references(
    class_names: list[str],
    rng: np.random.Generator,
    per_clip: int = 3,
    channel_db: float = 8.0,
) -> np.ndarray:
    """Unit-energy references from VTUAD class archives, random placement.

    The organisers varied "time trajectories, relative acoustic channel and
    energy". Placement covers the trajectory, ``channel_db`` applies a random
    smooth frequency-dependent gain, and the mixer supplies the energy.
    """
    levels = np.concatenate([collect_levels(name)[0] for name in class_names])
    return references_from_levels(levels, rng, per_clip, channel_db)


def references_from_levels(
    levels: np.ndarray,
    rng: np.random.Generator,
    per_clip: int = 3,
    channel_db: float = 8.0,
) -> np.ndarray:
    """Build unit-energy reference windows from a ``(n, 15, 25)`` dB tensor."""
    log_centred = np.log10(BAND_CENTRES) - np.log10(BAND_CENTRES).mean()
    references: list[np.ndarray] = []
    for clip in levels:
        for _ in range(per_clip):
            window = place_clip(clip, rng)
            if channel_db > 0:
                window = window + rng.uniform(-channel_db, channel_db) * log_centred[None, :]
            references.append(to_unit_energy(window))
    return np.stack(references)


def collect_strict_levels(archive: Path, min_minutes: int = 2) -> tuple[np.ndarray, list[str]]:
    """Band levels for the strict-interval archive (all vessel clips).

    The strict-interval release re-cuts every clip to the AIS-qualified interval,
    so it does not carry the 300 s of pre-passage audio that the class archives
    include for some clips. Clips shorter than ``min_minutes`` are dropped.
    """
    target = archive.parent / "strict_vessel"
    if not target.exists():
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(target)

    levels_list: list[np.ndarray] = []
    names: list[str] = []
    for path in sorted(target.rglob("*.wav")):
        levels = clip_levels(path)
        if len(levels) < min_minutes:
            continue
        window = np.full((MINUTES, N_BANDS), np.nan)
        take = min(len(levels), MINUTES)
        window[:take] = levels[:take]
        levels_list.append(window)
        names.append(path.name)

    if not levels_list:
        raise ValueError(f"no usable clips in {archive}")
    return np.stack(levels_list), names
