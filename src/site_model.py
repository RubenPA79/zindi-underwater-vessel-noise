"""Per-site background model used to measure excess energy in a window.

Sites differ by up to 20 dB in absolute level, so a window's loudness alone says
little. Matching a window to the site whose band spectrum it resembles gives a
local reference: the excess over that site's typical level is a direct estimate
of the added energy.

All statistics come from the training backgrounds; nothing is fitted on test.
"""

from __future__ import annotations

import numpy as np


class SiteMatcher:
    """Median band spectrum and typical total energy for each site."""

    def __init__(self) -> None:
        self.sites_: np.ndarray | None = None
        self.median_spectrum_: np.ndarray | None = None
        self.spectrum_scale_: np.ndarray | None = None
        self.median_total_: np.ndarray | None = None

    def fit(self, windows_db: np.ndarray, site: np.ndarray) -> "SiteMatcher":
        self.sites_ = np.unique(site)
        spectra, scales, totals = [], [], []
        for site_id in self.sites_:
            subset = windows_db[site == site_id]
            bands = np.nanmean(subset, axis=1)
            spectra.append(np.nanmedian(bands, axis=0))
            scales.append(np.nanstd(bands, axis=0) + 1e-6)
            totals.append(np.nanmedian((10.0 ** (subset / 10.0)).sum(axis=(1, 2))))
        self.median_spectrum_ = np.stack(spectra)
        self.spectrum_scale_ = np.stack(scales)
        self.median_total_ = np.asarray(totals)
        return self

    def transform(self, windows_db: np.ndarray) -> np.ndarray:
        """Per-site excess z-scores plus the best match and its share estimate."""
        if self.median_spectrum_ is None:
            raise RuntimeError("SiteMatcher must be fitted before transform")

        bands = np.nanmean(windows_db, axis=1)
        z_scores = (bands[:, None, :] - self.median_spectrum_[None]) / self.spectrum_scale_[None]
        excess = np.nanmean(z_scores, axis=2)
        best = np.argmax(excess, axis=1)

        total = np.nansum(10.0 ** (windows_db / 10.0), axis=(1, 2))
        reference_total = self.median_total_[best]
        share_estimate = 1.0 - reference_total / np.maximum(total, 1e-30)

        return np.column_stack(
            [
                excess,
                excess[np.arange(len(best)), best],
                best.astype(float),
                np.clip(share_estimate, -1.0, 1.0),
            ]
        ).astype(np.float32)

    @property
    def n_features(self) -> int:
        if self.median_spectrum_ is None:
            raise RuntimeError("SiteMatcher must be fitted before use")
        return len(self.sites_) + 3
