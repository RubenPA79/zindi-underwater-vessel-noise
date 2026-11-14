"""Train a baseline regressor on self-made mixtures and write a submission.

The test set is a real background plus a controlled passage addition, so the
target is exact for any mixture we build ourselves. References are real windows
from the archive (a permitted augmentation), added with a known energy gain so
``share_added_vessel`` can be computed with the official formula.

Validation is leave-one-site-out: a model must generalise to unseen sites, which
is the closest proxy we have for the hidden backgrounds.

Run from ``proyectos/underwater-vessel-noise`` as::

    python -m src.train_baseline
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import window_features
from src.layout import parse_columns, to_tensor

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROJECT_DIR = DATA_DIR.parent
WINDOWS = DATA_DIR / "bg_windows.npz"
SUBMISSION = PROJECT_DIR / "submission.csv"

RANDOM_SEED = 42
N_SAMPLES = 60_000
ZERO_FRACTION = 0.35
MISSING_CELL_RATE = 0.033
SHARE_MIN, SHARE_MAX = 1e-2, 0.79

PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": RANDOM_SEED,
    "num_threads": 0,
}
NUM_BOOST_ROUND = 3000
EARLY_STOPPING = 150


def sample_mixtures(
    backgrounds: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``n_samples`` synthetic mixtures with their exact share target."""
    n_backgrounds = len(backgrounds)
    background_index = rng.integers(0, n_backgrounds, n_samples)
    reference_index = rng.integers(0, n_backgrounds, n_samples)
    clash = reference_index == background_index
    reference_index[clash] = (reference_index[clash] + 1) % n_backgrounds

    background_energy = 10.0 ** (backgrounds[background_index].astype(np.float64) / 10.0)
    reference_energy = 10.0 ** (backgrounds[reference_index].astype(np.float64) / 10.0)
    reference_energy /= reference_energy.sum(axis=(1, 2), keepdims=True)

    share = np.zeros(n_samples)
    has_addition = rng.random(n_samples) >= ZERO_FRACTION
    share[has_addition] = 10.0 ** rng.uniform(np.log10(SHARE_MIN), np.log10(SHARE_MAX), has_addition.sum())

    # share = a / (b + a)  =>  a = share / (1 - share) * b
    background_total = background_energy.sum(axis=(1, 2))
    added_total = share / (1.0 - share) * background_total

    added = added_total[:, None, None] * reference_energy
    mixture = background_energy + added
    observed = rng.random(mixture.shape) >= MISSING_CELL_RATE

    added_observed = np.where(observed, added, 0.0).sum(axis=(1, 2))
    total_observed = np.where(observed, mixture, 0.0).sum(axis=(1, 2))
    target = np.divide(added_observed, total_observed, out=np.zeros_like(added_observed), where=total_observed > 0)

    with np.errstate(divide="ignore"):
        mixture_db = np.where(observed, 10.0 * np.log10(mixture), np.nan)
    return window_features(mixture_db), target, background_index


def load_test_features() -> tuple[list[str], np.ndarray]:
    test = pd.read_csv(DATA_DIR / "Test.csv")
    identifiers = test["ID"].tolist()
    features = list(test.columns[1:])
    minute_index, band_index = parse_columns(features)
    windows = to_tensor(test[features].to_numpy(dtype=float), minute_index, band_index)
    return identifiers, window_features(windows)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    archive = np.load(WINDOWS)
    backgrounds, site = archive["windows"], archive["site"]
    print(f"background windows: {len(backgrounds):,} from {len(np.unique(site))} sites")

    features, target, background_index = sample_mixtures(backgrounds, N_SAMPLES, rng)
    groups = site[background_index]
    print(f"synthetic mixtures: {features.shape[0]:,} x {features.shape[1]} features")
    print(f"target: mean {target.mean():.3f}, zeros {np.mean(target == 0):.1%}, max {target.max():.3f}")

    out_of_fold = np.full(len(target), np.nan)
    scores = {}
    best_iterations = []
    for held_out_site in np.unique(groups):
        train_mask = groups != held_out_site
        model = lgb.train(
            PARAMS,
            lgb.Dataset(features[train_mask], label=target[train_mask]),
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[lgb.Dataset(features[~train_mask], label=target[~train_mask])],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
        prediction = model.predict(features[~train_mask], num_iteration=model.best_iteration)
        out_of_fold[~train_mask] = prediction
        rmse = float(np.sqrt(np.mean((prediction - target[~train_mask]) ** 2)))
        scores[held_out_site] = 1.0 - rmse
        best_iterations.append(model.best_iteration)
        print(f"  site {held_out_site:>2}: rmse {rmse:.4f}  score {1 - rmse:.4f}  ({model.best_iteration} trees)")

    overall = float(np.sqrt(np.mean((out_of_fold - target) ** 2)))
    print(f"\nsynthetic OOF rmse {overall:.4f} (score {1 - overall:.4f} on synthetic targets only)")
    print(f"per-site score: mean {np.mean(list(scores.values())):.4f}, std {np.std(list(scores.values())):.4f}")

    n_trees = int(np.median(best_iterations))
    print(f"refitting on all data with {n_trees} trees")
    final_model = lgb.train(
        PARAMS,
        lgb.Dataset(features, label=target),
        num_boost_round=n_trees,
    )

    identifiers, test_features = load_test_features()
    prediction = np.clip(final_model.predict(test_features), 0.0, 1.0)
    print(
        f"\ntest prediction: mean {prediction.mean():.3f}, zeros {np.mean(prediction < 1e-6):.1%}, "
        f"p50 {np.median(prediction):.3f}, p95 {np.percentile(prediction, 95):.3f}"
    )

    submission = pd.DataFrame(
        {
            "ID": identifiers,
            "share_added_vessel": prediction,
            "score_added_vessel": np.clip(prediction, 0.0, 1.0),
        }
    )
    submission.to_csv(SUBMISSION, index=False)
    print(f"wrote {SUBMISSION.relative_to(PROJECT_DIR.parent)} ({len(submission)} rows)")

    oof = pd.DataFrame({"site": groups, "target": target, "prediction": out_of_fold})
    oof.to_csv(DATA_DIR / "oof_baseline.csv", index=False)
    print(f"wrote data/oof_baseline.csv")


if __name__ == "__main__":
    main()
