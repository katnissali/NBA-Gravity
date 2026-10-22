import unittest

import numpy as np

from src.models.train import (
    average_ranks,
    cross_validate_ols,
    evaluate,
    fit_ols,
    predict,
    split_players,
    standardize,
)


class ModelingTests(unittest.TestCase):
    def test_split_is_reproducible_and_disjoint(self):
        player_ids = [str(player_id) for player_id in range(20)]
        train_a, test_a = split_players(player_ids, test_size=0.2, seed=440)
        train_b, test_b = split_players(player_ids, test_size=0.2, seed=440)

        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)
        self.assertEqual(set(train_a).intersection(test_a), set())
        self.assertEqual(len(test_a), 4)

    def test_standardization_uses_training_statistics(self):
        train = np.array([[1.0, 10.0], [3.0, 14.0]])
        held_out = np.array([[100.0, 100.0]])
        standardized_train, standardized_test, means = standardize(train, held_out)

        np.testing.assert_allclose(means, [2.0, 12.0])
        np.testing.assert_allclose(standardized_train.mean(axis=0), [0.0, 0.0])
        np.testing.assert_allclose(standardized_test, [[98.0, 44.0]])

    def test_ols_recovers_known_linear_relation(self):
        features = np.array([[-1.0], [0.0], [1.0], [2.0]])
        target = 3.0 + 2.0 * features[:, 0]
        coefficients = fit_ols(features, target)

        np.testing.assert_allclose(coefficients, [3.0, 2.0])
        np.testing.assert_allclose(predict(features, coefficients), target)

    def test_average_ranks_assigns_average_rank_to_ties(self):
        np.testing.assert_allclose(average_ranks(np.array([5.0, 1.0, 1.0, 9.0])), [3, 1.5, 1.5, 4])

    def test_evaluation_reports_perfect_predictions(self):
        target = np.array([1.0, 2.0, 2.0, 4.0])
        metrics = evaluate(target, target.copy())

        self.assertEqual(metrics["MAE"], 0.0)
        self.assertEqual(metrics["R2"], 1.0)
        self.assertAlmostEqual(metrics["Spearman"], 1.0)

    def test_evaluation_reports_negative_test_r_squared_when_worse_than_mean(self):
        target = np.array([0.0, 1.0, 2.0])
        predictions = np.array([100.0, 100.0, 100.0])

        metrics = evaluate(target, predictions)
        self.assertLess(metrics["R2"], 0.0)
        self.assertIsNone(metrics["Spearman"])

    def test_cross_validation_is_reproducible_and_reports_fold_variation(self):
        features = np.arange(80, dtype=float).reshape(40, 2)
        features[:, 1] = np.sin(features[:, 1])
        target = 1.5 + 0.7 * features[:, 0] - 0.3 * features[:, 1]

        result_a = cross_validate_ols(features, target, folds=5, seed=440)
        result_b = cross_validate_ols(features, target, folds=5, seed=440)

        self.assertEqual(result_a, result_b)
        self.assertEqual(set(result_a), {"MAE", "R2", "Spearman"})
        self.assertTrue(all(mean >= 0 and sd >= 0 for mean, sd in result_a.values()))


if __name__ == "__main__":
    unittest.main()
