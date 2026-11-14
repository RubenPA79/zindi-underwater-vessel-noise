"""Tests for the window layout and the target definition.

Run from the repository root::

    python -m unittest discover -s proyectos/underwater-vessel-noise/tests -t proyectos/underwater-vessel-noise -v
"""

from __future__ import annotations

import unittest

import numpy as np

from src.layout import (
    BAND_CENTRES,
    MINUTES,
    N_BANDS,
    LayoutError,
    assert_full_grid,
    db_to_energy,
    parse_columns,
    synthetic_share,
    to_tensor,
)


def build_columns() -> list[str]:
    return [f"m{minute:02d}_TOL_{centre:.1f}" for minute in range(1, MINUTES + 1) for centre in BAND_CENTRES]


class BandGridTest(unittest.TestCase):
    def test_grid_shape_and_endpoints(self) -> None:
        self.assertEqual(len(BAND_CENTRES), N_BANDS)
        self.assertAlmostEqual(BAND_CENTRES[0], 31.6, places=6)
        self.assertTrue(7900 < BAND_CENTRES[-1] < 8000)

    def test_grid_is_strictly_increasing(self) -> None:
        self.assertTrue(np.all(np.diff(BAND_CENTRES) > 0))


class ParseColumnsTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        columns = build_columns()
        minutes, bands = parse_columns(columns)

        np.testing.assert_array_equal(minutes, np.repeat(np.arange(MINUTES), N_BANDS))
        np.testing.assert_array_equal(bands, np.tile(np.arange(N_BANDS), MINUTES))

    def test_accepts_integer_and_decimal_frequencies(self) -> None:
        _, bands = parse_columns(["m01_TOL_31.6", "m01_TOL_100", "m01_TOL_7943"])
        self.assertEqual(list(bands), [0, 5, 24])

    def test_rejects_unknown_frequency(self) -> None:
        with self.assertRaises(LayoutError):
            parse_columns(["m01_TOL_9999"])

    def test_rejects_bad_names(self) -> None:
        for name in ("TOL_31.6", "m16_TOL_31.6", "m00_TOL_31.6", "m1-TOL-31.6"):
            with self.subTest(name=name), self.assertRaises(LayoutError):
                parse_columns([name])

    def test_rejects_duplicates(self) -> None:
        with self.assertRaises(LayoutError):
            parse_columns(["m01_TOL_31.6", "m01_TOL_31.60"])


class TensorTest(unittest.TestCase):
    def test_places_each_value_in_its_cell(self) -> None:
        columns = build_columns()
        minutes, bands = parse_columns(columns)
        values = np.array([np.arange(len(columns), dtype=float)])

        tensor = to_tensor(values, minutes, bands)

        self.assertEqual(tensor.shape, (1, MINUTES, N_BANDS))
        self.assertEqual(tensor[0, 3, 7], float(3 * N_BANDS + 7))

    def test_missing_cells_stay_nan(self) -> None:
        columns = ["m01_TOL_31.6", "m02_TOL_100"]
        minutes, bands = parse_columns(columns)
        tensor = to_tensor(np.array([[1.0, 2.0]]), minutes, bands)

        self.assertEqual(tensor[0, 0, 0], 1.0)
        self.assertEqual(tensor[0, 1, 5], 2.0)
        self.assertTrue(np.isnan(tensor[0, 0, 5]))
        self.assertEqual(int(np.isnan(tensor[0]).sum()), MINUTES * N_BANDS - 2)


class FullGridTest(unittest.TestCase):
    def test_accepts_a_complete_grid(self) -> None:
        minutes, bands = parse_columns(build_columns())
        assert_full_grid(minutes, bands)

    def test_rejects_missing_cells(self) -> None:
        columns = build_columns()
        for position in (0, 17, len(columns) - 1):
            with self.subTest(position=position), self.assertRaises(LayoutError):
                minutes, bands = parse_columns(columns[:position] + columns[position + 1 :])
                assert_full_grid(minutes, bands)

    def test_rejects_duplicated_cells(self) -> None:
        minutes, bands = parse_columns(build_columns())
        with self.assertRaises(LayoutError):
            assert_full_grid(minutes[:-1], bands[:-1])


class EnergyTest(unittest.TestCase):
    def test_decades(self) -> None:
        self.assertAlmostEqual(float(db_to_energy(0.0)), 1.0, places=9)
        self.assertAlmostEqual(float(db_to_energy(10.0)), 10.0, places=9)

    def test_nan_is_preserved(self) -> None:
        self.assertTrue(np.isnan(db_to_energy(np.nan)))


class SyntheticShareTest(unittest.TestCase):
    def test_official_example(self) -> None:
        """2 units added to a mixture totalling 10 gives a target of 0.20."""
        background_db = np.full((MINUTES, N_BANDS), 10 * np.log10(8.0))

        self.assertAlmostEqual(synthetic_share(background_db, 2.0), 0.2, places=12)

    def test_no_addition_is_zero(self) -> None:
        background_db = np.full((MINUTES, N_BANDS), 55.0)

        self.assertEqual(synthetic_share(background_db, 0.0), 0.0)

    def test_db_zero_is_not_no_addition(self) -> None:
        """A 0 dB reference carries unit energy; only 0 energy means silence."""
        background_db = np.full((MINUTES, N_BANDS), 10 * np.log10(1.0))

        self.assertAlmostEqual(synthetic_share(background_db, db_to_energy(0.0)), 0.5, places=12)

    def test_uses_linear_energy_not_decibels(self) -> None:
        """Adding the same energy as the background gives a share of one half."""
        background_db = np.full((MINUTES, N_BANDS), 60.0)

        self.assertAlmostEqual(synthetic_share(background_db, db_to_energy(60.0)), 0.5, places=12)

    def test_missing_cells_are_excluded_from_both_sums(self) -> None:
        background_db = np.full((MINUTES, N_BANDS), 10 * np.log10(8.0))
        background_db[0, :] = np.nan

        self.assertAlmostEqual(synthetic_share(background_db, 2.0), 0.2, places=12)

    def test_louder_addition_raises_the_share(self) -> None:
        background_db = np.full((MINUTES, N_BANDS), 10 * np.log10(100.0))
        quiet = synthetic_share(background_db, 1.0)
        loud = synthetic_share(background_db, 100.0)

        self.assertLess(quiet, loud)
        self.assertAlmostEqual(loud, 0.5, places=12)

    def test_raises_when_nothing_is_observed(self) -> None:
        with self.assertRaises(LayoutError):
            synthetic_share(np.full((MINUTES, N_BANDS), np.nan), 50.0)


if __name__ == "__main__":
    unittest.main()
