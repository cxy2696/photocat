import unittest

import numpy as np

from photocat.data.dataset import MAX_TOKENS, Featurizer, ph_bucket
from photocat.data.synth import generate_dataset, parse_formula


class DataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = generate_dataset(n=500, seed=3)

    def test_formula_parser(self):
        self.assertEqual(parse_formula("TiO2"), [("Ti", 1.0), ("O", 2.0)])
        self.assertEqual(parse_formula("Bi2WO6"), [("Bi", 2.0), ("W", 1.0), ("O", 6.0)])

    def test_split_has_no_group_leakage(self):
        train = {(material.formula, material.dopant) for material in self.bundle.train}
        test = {(material.formula, material.dopant) for material in self.bundle.test}
        self.assertFalse(train & test)
        self.assertTrue(set(material.family for material in self.bundle.test))
        self.assertEqual({material.family for material in self.bundle.ood}, {"oxynitride", "sulfide"})

    def test_featurizer_shapes_and_masks(self):
        featurizer = Featurizer.fit(self.bundle.train)
        batch = featurizer.transform(self.bundle.test[:32])
        self.assertEqual(batch.token_type.shape, (32, MAX_TOKENS))
        self.assertEqual(batch.node_features.shape[0], 32)
        self.assertTrue(batch.token_mask[:, 0].all())
        self.assertTrue((batch.token_type[:, 0] == 1).all())
        sums = batch.adjacency.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-9)
        # Padding tokens must not receive graph bias.
        self.assertTrue(np.all(batch.pair_bias[~batch.token_mask] == 0.0))

    def test_generation_is_deterministic(self):
        other = generate_dataset(n=200, seed=9)
        again = generate_dataset(n=200, seed=9)
        self.assertEqual([m.eg for m in other.materials], [m.eg for m in again.materials])
        self.assertEqual(ph_bucket(7.4), "3")


if __name__ == "__main__":
    unittest.main()
