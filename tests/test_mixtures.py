"""Tests for the synthetic mixture builder.

Run from the repository root::

    python -m unittest discover -s proyectos/underwater-vessel-noise/tests -t proyectos/underwater-vessel-noise -v
"""

from __future__ import annotations

import unittest

import numpy as np

from src import mixtures
from src.audio import bell_envelope, passage_window, unit_energy_reference
from src.layout import MINUTES, N_BANDS, synthetic_share


class ReferenceTest(unittest.TestCase):
    def test_unit_energy(self) -> None:
        levels = np.linspace(80.0, 50.0, N_BANDS)
        reference = unit_energy_reference(passage_window(levels))

        self.assertEqual(reference.shape, (MINUTES, N_BANDS))
        self.assertAlmostEqual(float(reference.sum()), 1.0, places=12)

    def test_bell_envelope_peaks_in_the_middle(self) -> None:
        envelope = bell_envelope()

        self.assertEqual(envelope.shape, (MINUTES,))
        self.assertAlmostEqual(float(envelope.max()), 1.0, places=12)
        self.assertEqual(int(envelope.argmax()), 7)
        self.assertLess(envelope[0], envelope[7])

    def test_build_references_shape(self) -> None:
        levels = np.full((4, N_BANDS), 70.0)
        rng = np.random.default_rng(0)

        references = mixtures.build_references(levels, rng, variants=3)

        self.assertEqual(references.shape, (12, MINUTES, N_BANDS))
        np.testing.assert_allclose(references.sum(axis=(1, 2)), 1.0, rtol=1e-12)


class ObservedShareTest(unittest.TestCase):
    def test_matches_the_official_formula(self) -> None:
        """The mixture target must equal ``layout.synthetic_share`` exactly."""
        rng = np.random.default_rng(3)
        background_db = rng.uniform(40.0, 100.0, size=(5, MINUTES, N_BANDS))
        added = rng.uniform(1e-3, 1e3, size=(5, MINUTES, N_BANDS))
        observed = rng.random(background_db.shape) > 0.2

        vectorised = mixtures.observed_share(10.0 ** (background_db / 10.0), added, observed)
        expected = np.array(
            [synthetic_share(background_db[i], added[i], valid=observed[i]) for i in range(len(background_db))]
        )

        np.testing.assert_allclose(vectorised, expected, rtol=1e-12)

    def test_fully_missing_window_scores_zero(self) -> None:
        background = np.ones((MINUTES, N_BANDS))
        added = np.ones((MINUTES, N_BANDS))
        observed = np.zeros((MINUTES, N_BANDS), dtype=bool)

        share = mixtures.observed_share(background[None], added[None], observed[None])

        self.assertEqual(float(share[0]), 0.0)


class SampleMixturesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backgrounds = np.full((3, MINUTES, N_BANDS), 80.0)
        self.references = np.full((1, MINUTES, N_BANDS), 1.0 / (MINUTES * N_BANDS))

    def test_no_addition_gives_zero_target(self) -> None:
        rng = np.random.default_rng(1)

        features, target, index = mixtures.sample_mixtures(
            self.backgrounds, self.references, 100, rng, zero_fraction=1.0
        )

        self.assertTrue(np.all(target == 0.0))
        self.assertEqual(features.shape[0], 100)
        self.assertTrue(np.all((index >= 0) & (index < len(self.backgrounds))))

    def test_target_equals_the_gain_without_missing_cells(self) -> None:
        """With every cell observed, the target is exactly the chosen share."""
        original = mixtures.MISSING_CELL_RATE
        mixtures.MISSING_CELL_RATE = 0.0
        try:
            rng = np.random.default_rng(7)
            _, target, _ = mixtures.sample_mixtures(self.backgrounds, self.references, 2000, rng, zero_fraction=0.0)
        finally:
            mixtures.MISSING_CELL_RATE = original

        non_zero = target[target > 0]
        self.assertGreater(len(non_zero), 1900)
        self.assertTrue(np.all(non_zero >= mixtures.SHARE_MIN * 0.999))
        self.assertTrue(np.all(non_zero <= mixtures.SHARE_MAX * 1.001))


if __name__ == "__main__":
    unittest.main()
