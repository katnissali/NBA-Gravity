"""Fit and evaluate the planned six-factor OLS gravity model."""

import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "player_gravity_model_data.csv"
PREDICTORS = (
    "OFF_RATING_DIFF",
    "EFG_DIFF_PP",
    "AST_PER_GAME",
    "TIMEOUTS_PER_36",
    "CLUTCH_PTS_PER_36",
    "OWS",
)


def load_complete_cases(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load players with all six predictors and a valid numeric GRAV target."""
    player_ids: list[str] = []
    features: list[list[float]] = []
    target: list[float] = []

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"PLAYER_ID", "GRAV", "ALL_SIX_AVAILABLE", *PREDICTORS}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        for row in reader:
            if row["ALL_SIX_AVAILABLE"].strip().lower() != "true":
                continue
            try:
                row_target = float(row["GRAV"])
                row_features = [float(row[name]) for name in PREDICTORS]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid complete-case values for player {row.get('PLAYER_ID')}"
                ) from exc
            if not np.isfinite(row_target) or not np.isfinite(row_features).all():
                raise ValueError(
                    f"Non-finite complete-case values for player {row['PLAYER_ID']}"
                )
            player_ids.append(row["PLAYER_ID"])
            features.append(row_features)
            target.append(row_target)

    if len(player_ids) != len(set(player_ids)):
        raise ValueError("Complete-case dataset contains duplicate PLAYER_ID values")
    if len(player_ids) < 2:
        raise ValueError("At least two complete-case players are required")

    return player_ids, np.asarray(features, dtype=float), np.asarray(target, dtype=float)


def split_players(
    player_ids: list[str],
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic train/test row indices, shuffled at the player level."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    test_count = int(round(len(player_ids) * test_size))
    if test_count < 1 or test_count >= len(player_ids):
        raise ValueError("Split must leave at least one player in both sets")

    indices = np.random.default_rng(seed).permutation(len(player_ids))
    return indices[test_count:], indices[:test_count]


def standardize(
    train_features: np.ndarray,
    other_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize using training-set means and population standard deviations."""
    means = train_features.mean(axis=0)
    scales = train_features.std(axis=0)
    if not np.isfinite(scales).all() or np.any(scales == 0):
        raise ValueError("Cannot standardize a predictor with zero variance")
    return (train_features - means) / scales, (other_features - means) / scales, means


def fit_ols(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit an intercept and OLS coefficients using a numerically stable solver."""
    design = np.column_stack((np.ones(len(features)), features))
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients


def predict(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(features)), features))
    return design @ coefficients


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks, handling ties."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def evaluate(
    target: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float | None]:
    """Calculate test MAE, R-squared, and Spearman rank correlation."""
    errors = target - predictions
    total_sum_squares = np.sum((target - target.mean()) ** 2)
    if total_sum_squares == 0:
        raise ValueError("Test targets have zero variance; R-squared is undefined")

    target_ranks = average_ranks(target)
    prediction_ranks = average_ranks(predictions)
    if np.std(prediction_ranks) == 0:
        rank_correlation = None
    else:
        rank_correlation = float(np.corrcoef(target_ranks, prediction_ranks)[0, 1])
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "R2": float(1 - np.sum(errors**2) / total_sum_squares),
        "Spearman": rank_correlation,
    }


def cross_validate_ols(
    features: np.ndarray,
    target: np.ndarray,
    folds: int = 5,
    seed: int = 440,
) -> dict[str, tuple[float, float]]:
    """Return fold-mean and fold standard deviation for training-set OLS CV."""
    if folds < 2 or folds > len(target):
        raise ValueError("folds must be between 2 and the number of observations")

    indices = np.random.default_rng(seed).permutation(len(target))
    validation_folds = np.array_split(indices, folds)
    fold_metrics: dict[str, list[float]] = {
        "MAE": [],
        "R2": [],
        "Spearman": [],
    }

    for fold_number, validation_indices in enumerate(validation_folds):
        training_indices = np.concatenate(
            [fold for i, fold in enumerate(validation_folds) if i != fold_number]
        )
        train_features, validation_features, _ = standardize(
            features[training_indices],
            features[validation_indices],
        )
        coefficients = fit_ols(train_features, target[training_indices])
        metrics = evaluate(
            target[validation_indices],
            predict(validation_features, coefficients),
        )
        for name, value in metrics.items():
            if value is None:
                raise ValueError(f"{name} is undefined in cross-validation fold")
            fold_metrics[name].append(value)

    return {
        name: (float(np.mean(values)), float(np.std(values, ddof=1)))
        for name, values in fold_metrics.items()
    }


def _calibrated_single_predictor(
    train_column: np.ndarray,
    test_column: np.ndarray,
    train_target: np.ndarray,
) -> np.ndarray:
    design = np.column_stack((np.ones(len(train_column)), train_column))
    coefficients, _, _, _ = np.linalg.lstsq(design, train_target, rcond=None)
    return coefficients[0] + test_column * coefficients[1]


def run_model(path: Path = DATA_PATH, test_size: float = 0.2, seed: int = 440) -> None:
    player_ids, features, target = load_complete_cases(path)
    train_indices, test_indices = split_players(player_ids, test_size, seed)
    train_raw = features[train_indices]
    test_raw = features[test_indices]
    train_target = target[train_indices]
    test_target = target[test_indices]

    train_features, test_features, means = standardize(train_raw, test_raw)
    scales = train_raw.std(axis=0)
    coefficients = fit_ols(train_features, train_target)
    predictions = predict(test_features, coefficients)

    print(f"Dataset: {path}")
    print(f"Complete-case players: {len(player_ids)}")
    print(f"Train/test players: {len(train_indices)}/{len(test_indices)}")
    print(f"Random seed: {seed}")
    cv_metrics = cross_validate_ols(
        train_raw,
        train_target,
        folds=5,
        seed=seed,
    )
    print("\nFive-fold CV on training players only (mean +/- fold SD):")
    for name, (mean, standard_deviation) in cv_metrics.items():
        print(f"  {name}: {mean:.10f} +/- {standard_deviation:.10f}")

    print("\nOLS model (standardized predictors):")
    print(f"  Intercept: {coefficients[0]:.10f}")
    for name, coefficient in zip(PREDICTORS, coefficients[1:]):
        print(f"  {name}: {coefficient:.10f}")
    print("\nHeld-out OLS metrics:")
    _print_metrics(evaluate(test_target, predictions))

    train_index_score = train_features.mean(axis=1)
    test_index_score = test_features.mean(axis=1)
    benchmark_coefficients = fit_ols(train_index_score[:, None], train_target)
    benchmark_predictions = (
        benchmark_coefficients[0] + benchmark_coefficients[1] * test_index_score
    )
    print("\nHeld-out calibrated equal-weight benchmark:")
    _print_metrics(evaluate(test_target, benchmark_predictions))

    print("\nHeld-out single-predictor OLS baselines:")
    for column, name in enumerate(PREDICTORS):
        baseline_predictions = _calibrated_single_predictor(
            train_features[:, column],
            test_features[:, column],
            train_target,
        )
        print(f"  {name}:")
        _print_metrics(evaluate(test_target, baseline_predictions), indent="    ")

    print("\nTraining-only standardization parameters:")
    for name, mean, scale in zip(PREDICTORS, means, scales):
        print(f"  {name}: mean={mean:.10f}, std={scale:.10f}")


def _print_metrics(metrics: dict[str, float | None], indent: str = "  ") -> None:
    for name, value in metrics.items():
        formatted = "undefined (constant predictions)" if value is None else f"{value:.10f}"
        print(f"{indent}{name}: {formatted}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Input merged CSV")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out fraction")
    parser.add_argument("--seed", type=int, default=440, help="Random split seed")
    args = parser.parse_args()
    run_model(args.data, args.test_size, args.seed)


if __name__ == "__main__":
    main()
