"""Train on mixtures built from real vessel passages and write a submission.

Same protocol as ``train_baseline`` but the added energy comes from HearMyShip
recordings converted to third-octave bands, which gives the model a spectral and
temporal signature a plain background window does not have.

Run from ``proyectos/underwater-vessel-noise`` as::

    python -m src.train_passage
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import window_features
from src.layout import N_BANDS, parse_columns, to_tensor
from src.mixtures import (
    DEFAULT_ZERO_FRACTION,
    SHARE_MAX,
    SHARE_MIN,
    background_references,
    build_references,
    flat_reference,
    sample_mixture_windows,
)
from src.site_model import SiteMatcher
from src.train_baseline import EARLY_STOPPING, NUM_BOOST_ROUND, PARAMS
from src.vtuad import collect_strict_levels, references_from_levels, vtuad_references

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROJECT_DIR = DATA_DIR.parent
SUBMISSION = PROJECT_DIR / "submission_passage.csv"

RANDOM_SEED = 2024
N_SAMPLES = 60_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=N_SAMPLES)
    parser.add_argument("--rounds", type=int, default=NUM_BOOST_ROUND)
    parser.add_argument("--early-stopping", type=int, default=EARLY_STOPPING)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--zeros", type=float, default=DEFAULT_ZERO_FRACTION, help="fraction of windows with no addition")
    parser.add_argument("--share-min", type=float, default=SHARE_MIN)
    parser.add_argument("--share-max", type=float, default=SHARE_MAX)
    parser.add_argument("--sa-refs", type=int, default=40, help="local references per site (0 disables)")
    parser.add_argument("--hear-refs", type=int, default=2000, help="cap on HearMyShip reference windows")
    parser.add_argument("--raw", action="store_true", help="also feed the raw 15x25 grid to the model")
    parser.add_argument(
        "--vtuad",
        nargs="*",
        default=None,
        help="use VTUAD classes (e.g. pilotvessel tanker sailing) instead of HearMyShip references",
    )
    parser.add_argument(
        "--vtuad-strict",
        type=Path,
        default=None,
        help="path to the concatenated vtuad_strict_wav_vessel.zip (cleanest references)",
    )
    parser.add_argument("--out", type=Path, default=SUBMISSION)
    return parser.parse_args(argv)


def load_reference_levels() -> np.ndarray:
    frame = pd.read_csv(DATA_DIR / "vessel_levels.csv")
    columns = [f"b{band:02d}" for band in range(N_BANDS)]
    return frame[columns].to_numpy(dtype=np.float64)


def build_features(windows: np.ndarray, matcher: SiteMatcher, raw: bool = False) -> np.ndarray:
    """Aggregate features, per-site excess and optionally the raw 15x25 grid.

    The aggregates compress fifteen minutes into five numbers per band; the raw
    grid keeps the minute-by-minute pattern, which is where a short passage
    lives.
    """
    parts = [window_features(windows), matcher.transform(windows)]
    if raw:
        parts.append(windows.reshape(len(windows), -1))
    return np.hstack(parts)


def load_test_features(matcher: SiteMatcher, raw: bool = False) -> tuple[list[str], np.ndarray]:
    test = pd.read_csv(DATA_DIR / "Test.csv")
    identifiers = test["ID"].tolist()
    features = list(test.columns[1:])
    minute_index, band_index = parse_columns(features)
    windows = to_tensor(test[features].to_numpy(dtype=float), minute_index, band_index)
    return identifiers, build_features(windows, matcher, raw)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    params = dict(PARAMS)
    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate

    rng = np.random.default_rng(RANDOM_SEED)
    archive = np.load(DATA_DIR / "bg_windows.npz")
    backgrounds, site = archive["windows"], archive["site"]

    if args.vtuad_strict is not None:
        strict_levels, strict_names = collect_strict_levels(args.vtuad_strict)
        references = references_from_levels(strict_levels, rng)
        print(
            f"VTUAD strict references: {len(references)} from {len(strict_names)} clips "
            f"({args.vtuad_strict.name})",
            flush=True,
        )
    elif args.vtuad:
        references = vtuad_references(args.vtuad, rng)
        print(f"VTUAD references: {len(references)} from classes {args.vtuad}", flush=True)
    else:
        levels = load_reference_levels()
        if len(levels) == 0:
            raise SystemExit("no vessel references yet; run src.fetch_references first")
        hear = build_references(levels, rng)
        if args.hear_refs < len(hear):
            hear = hear[rng.choice(len(hear), args.hear_refs, replace=False)]
        references = [hear, np.stack([flat_reference(level) for level in levels])]
        if args.sa_refs > 0:
            local = background_references(backgrounds, site, rng, per_site=args.sa_refs)
            references.append(local)
            print(f"local references from the archive: {len(local)}", flush=True)
        references = np.concatenate(references)
        print(f"vessel references: {len(references)} from {len(levels)} vessels", flush=True)
    print(f"background windows: {len(backgrounds):,}", flush=True)

    mixture_db, target, background_index = sample_mixture_windows(
        backgrounds, references, args.samples, rng, args.zeros, (args.share_min, args.share_max)
    )
    matcher = SiteMatcher().fit(backgrounds, site)
    features = build_features(mixture_db, matcher, args.raw)
    groups = site[background_index]
    print(f"mixtures: {features.shape[0]:,} x {features.shape[1]} features, lr {params['learning_rate']}", flush=True)
    print(
        f"prior: zeros {args.zeros:.0%}, added share in [{args.share_min}, {args.share_max}]",
        flush=True,
    )
    print(f"target: mean {target.mean():.3f}, zeros {np.mean(target == 0):.1%}, max {target.max():.3f}", flush=True)

    out_of_fold = np.full(len(target), np.nan)
    best_iterations = []
    for held_out_site in np.unique(groups):
        train_mask = groups != held_out_site
        model = lgb.train(
            params,
            lgb.Dataset(features[train_mask], label=target[train_mask]),
            num_boost_round=args.rounds,
            valid_sets=[lgb.Dataset(features[~train_mask], label=target[~train_mask])],
            callbacks=[lgb.early_stopping(args.early_stopping, verbose=False)],
        )
        prediction = model.predict(features[~train_mask], num_iteration=model.best_iteration)
        out_of_fold[~train_mask] = prediction
        rmse = float(np.sqrt(np.mean((prediction - target[~train_mask]) ** 2)))
        best_iterations.append(model.best_iteration)
        print(f"  site {held_out_site:>2}: rmse {rmse:.4f}  ({model.best_iteration} trees)", flush=True)

    overall = float(np.sqrt(np.mean((out_of_fold - target) ** 2)))
    constant = float(np.sqrt(np.mean((target - target.mean()) ** 2)))
    verdict = "BETTER" if overall < constant else "NO GAIN"
    print(f"\nsynthetic OOF rmse {overall:.4f} vs constant {constant:.4f} ({verdict})", flush=True)
    presence = float(np.mean((out_of_fold > 0.25) == (target > 0.25)))
    print(f"presence accuracy at 0.25: {presence:.3f}", flush=True)

    n_trees = max(int(np.median(best_iterations)), 20)
    print(f"refitting on all data with {n_trees} trees", flush=True)
    final_model = lgb.train(params, lgb.Dataset(features, label=target), num_boost_round=n_trees)

    identifiers, test_features = load_test_features(matcher, args.raw)
    prediction = np.clip(final_model.predict(test_features), 0.0, 1.0)

    # A real background window has no controlled addition, so its true share is
    # zero. Any mass the model puts above zero here is pure false positive and
    # will be paid for on the hidden test.
    background_matrix = build_features(backgrounds, matcher, args.raw)
    background_prediction = np.clip(final_model.predict(background_matrix), 0.0, 1.0)
    print(
        f"\nbackground check (true share = 0): mean {background_prediction.mean():.3f}, "
        f"p50 {np.median(background_prediction):.3f}, p90 {np.percentile(background_prediction, 90):.3f}, "
        f"false positives >0.25: {np.mean(background_prediction > 0.25):.1%}",
        flush=True,
    )
    print(
        f"\ntest prediction: mean {prediction.mean():.3f}, zeros {np.mean(prediction < 1e-6):.1%}, "
        f"p50 {np.median(prediction):.3f}, p95 {np.percentile(prediction, 95):.3f}",
        flush=True,
    )
    counts, edges = np.histogram(prediction, bins=10, range=(0.0, 0.5))
    peak = max(counts.max(), 1)
    print("test prediction histogram (0..0.5):", flush=True)
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        print(f"  {left:.2f}..{right:.2f} {count:4d} {'#' * int(40 * count / peak)}", flush=True)

    submission = pd.DataFrame(
        {
            "ID": identifiers,
            "share_added_vessel": prediction,
            "score_added_vessel": prediction,
        }
    )
    submission.to_csv(args.out, index=False)
    print(f"wrote {args.out.name} ({len(submission)} rows)", flush=True)


if __name__ == "__main__":
    main()
