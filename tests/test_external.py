import unittest
from pathlib import Path

import numpy as np

from photocat.data.dataset import Featurizer
from photocat.data.external import (
    assign_group_splits,
    load_ees_her,
    load_matbench_gap,
    split_summary,
)
from photocat.models.kernel import KernelRidge, group_cross_validate
from photocat.training import TargetScaler, TrainConfig, Trainer

FIXTURES = Path(__file__).parent / "fixtures"


class ExternalDataTest(unittest.TestCase):
    def test_matbench_fixture_parses(self):
        materials = load_matbench_gap(FIXTURES / "matbench_sample.json")
        self.assertGreater(len(materials), 30)
        self.assertTrue(all(np.isnan(m.her) for m in materials))
        self.assertTrue(all(m.family == "unknown" for m in materials))
        assign_group_splits(
            materials,
            key=lambda material: frozenset(s for s, _, _ in material.elements),
            seed=1,
        )
        counts = split_summary(materials)
        self.assertGreater(counts["train"], 0)
        self.assertGreater(counts["test"], 0)

    def test_ees_fixture_loads(self):
        materials = load_ees_her(FIXTURES / "ees_sample.csv")
        self.assertEqual(len(materials), 8)
        self.assertEqual(materials[0].formula, "TiO2")
        self.assertAlmostEqual(materials[0].her, 75.0)
        self.assertEqual(materials[0].cocatalyst, "Pt")
        assign_group_splits(materials, key=lambda material: material.reference, seed=2)
        self.assertGreater(split_summary(materials)["train"], 0)

    def test_masked_target_training(self):
        materials = load_matbench_gap(FIXTURES / "matbench_sample.json")
        assign_group_splits(
            materials,
            key=lambda material: frozenset(s for s, _, _ in material.elements),
            seed=3,
        )
        train = [m for m in materials if m.split == "train"]
        val = [m for m in materials if m.split == "val"]
        featurizer = Featurizer.fit(train)
        scaler = TargetScaler.fit(featurizer.transform(train).targets)
        self.assertTrue(scaler.observed[0])
        self.assertFalse(scaler.observed[1])
        self.assertFalse(scaler.observed[2])
        from photocat.data.elements import ELEMENT_IDS
        from photocat.models.transformer import GraphAugmentedTransformer

        model = GraphAugmentedTransformer(
            max(ELEMENT_IDS.values()) + 1,
            [featurizer.vocab_size(field) for field in featurizer.ctx_vocab],
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=48,
            gnn_dim=16,
            seed=0,
        )
        result = Trainer(TrainConfig(epochs=1, verbose=False)).train(
            model, featurizer, train, val
        )
        self.assertEqual(len(result.history.epochs), 1)
        self.assertTrue(np.isfinite(result.history.best_val_loss))

    def test_kernel_ridge_group_cv(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 5))
        y = x[:, 0] * 2 - x[:, 1] + 0.1 * rng.normal(size=40)
        groups = [f"paper-{i % 8}" for i in range(40)]
        predictions, folds = group_cross_validate(x, y, groups, folds=4, seed=0)
        self.assertTrue(np.isfinite(predictions).all())
        self.assertGreater(np.corrcoef(y, predictions)[0, 1], 0.8)
        self.assertEqual(len(folds), 4)
        model = KernelRidge().fit(x, y)
        self.assertTrue(np.isfinite(model.predict(x)).all())


if __name__ == "__main__":
    unittest.main()
