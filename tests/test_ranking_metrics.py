import unittest

import numpy as np

from photocat.data.fingerprint import composition_fingerprint, pairwise_similarity
from photocat.data.synth import generate_dataset
from photocat.metrics import (
    conformal_quantile,
    enrichment_factor,
    pearson,
    rmse,
    spearman,
    top_k_recall,
)
from photocat.ranking import RankConfig, pareto_mask, screen_candidates


class RankingMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = generate_dataset(n=200, seed=11)

    def test_fingerprint_similarity(self):
        material = self.bundle.materials[0]
        fingerprint = composition_fingerprint(material)
        self.assertAlmostEqual(pairwise_similarity(fingerprint, fingerprint), 1.0, places=6)
        different = self.bundle.materials[-1]
        other = composition_fingerprint(different)
        self.assertLessEqual(pairwise_similarity(fingerprint, other), 1.0)

    def test_screening_filters(self):
        pool = self.bundle.test[:20]
        predictions = {
            "eg": np.full(20, 2.0),
            "her": np.full(20, 10.0),
            "stability": np.full(20, 0.8),
            "eg_std": np.full(20, 0.1),
            "her_std": np.full(20, 0.5),
        }
        candidates, funnel = screen_candidates(
            pool, predictions, self.bundle.train, RankConfig(top_k=5)
        )
        self.assertLessEqual(len(candidates), 5)
        self.assertGreaterEqual(funnel["physical"], len(candidates))
        for candidate in candidates:
            self.assertLess(candidate.similarity, 0.8)

    def test_pareto_mask(self):
        objectives = np.asarray([[1.0, 1.0], [2.0, 2.0], [0.5, 0.5]])
        mask = pareto_mask(objectives)
        self.assertTrue(mask[1])
        self.assertFalse(mask[0])
        self.assertFalse(mask[2])

    def test_metrics(self):
        truth = np.arange(50.0)
        prediction = truth + np.sin(truth)
        self.assertLess(rmse(truth, prediction), 1.1)
        self.assertGreater(pearson(truth, prediction), 0.99)
        self.assertGreater(spearman(truth, prediction), 0.99)
        labels = np.zeros(50, dtype=bool)
        labels[[48, 49]] = True
        self.assertEqual(top_k_recall(labels, prediction, 2), 1.0)
        self.assertGreater(enrichment_factor(labels, prediction, 2), 1.0)
        residuals = np.abs(np.random.default_rng(0).normal(0, 0.2, 1000))
        quantile = conformal_quantile(residuals, 0.9)
        self.assertGreater(quantile, 0.2)


if __name__ == "__main__":
    unittest.main()
