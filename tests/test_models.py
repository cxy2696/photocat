import tempfile
import unittest
from pathlib import Path

import numpy as np

from photocat.data.dataset import Featurizer
from photocat.data.synth import generate_dataset
from photocat.models.transformer import GraphAugmentedTransformer
from photocat.training import TargetScaler, TrainConfig, Trainer, forward_batches


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = generate_dataset(n=400, seed=7)
        cls.featurizer = Featurizer.fit(cls.bundle.train)
        cls.batch = cls.featurizer.transform(cls.bundle.test[:24])
        cls.ctx_sizes = [cls.featurizer.vocab_size(field) for field in cls.featurizer.ctx_vocab]
        cls.model = GraphAugmentedTransformer(
            64, cls.ctx_sizes, d_model=32, n_heads=4, n_layers=2, d_ff=48, gnn_dim=16, seed=2
        )

    def test_forward_shapes_and_attention(self):
        output = self.model.forward(self.batch, return_attention=True)
        self.assertEqual(output.eg.shape, (24,))
        self.assertEqual(output.her.shape, (24,))
        self.assertEqual(output.stability.shape, (24,))
        attention = output.attention[0]
        self.assertEqual(attention.shape, (24, 4, 14, 14))
        np.testing.assert_allclose(attention.sum(axis=-1), 1.0, atol=1e-9)
        leak = np.where(self.batch.token_mask[:, None, None, :], 0.0, attention).max()
        self.assertLess(leak, 1e-8)

    def test_all_parameters_receive_gradient(self):
        output = self.model.forward(self.batch)
        loss = (
            (output.eg * output.eg).mean()
            + (output.her * output.her).mean()
            + (output.stability * output.stability).mean()
        )
        loss.backward()
        for name, parameter in self.model.named_params().items():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(np.abs(parameter.grad).sum(), 0.0, name)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            self.model.save(str(path))
            model = GraphAugmentedTransformer.from_config(self.model.config(), seed=0)
            model.load(str(path))
            original = self.model.forward(self.batch).predict()
            restored = model.forward(self.batch).predict()
            np.testing.assert_allclose(original, restored, rtol=1e-10, atol=1e-10)

    def test_training_decreases_loss(self):
        model = GraphAugmentedTransformer(
            64, self.ctx_sizes, d_model=32, n_heads=4, n_layers=1, d_ff=48, gnn_dim=16, seed=3
        )
        trainer = Trainer(TrainConfig(epochs=3, batch_size=64, verbose=False, seed=1))
        result = trainer.train(
            model, self.featurizer, self.bundle.train[:250], self.bundle.val
        )
        losses = [row["train_loss"] for row in result.history.epochs]
        self.assertLess(losses[-1], losses[0])
        raw = forward_batches(model, self.batch)["raw"]
        self.assertEqual(np.asarray(raw).shape, (24, 3))
        scaler = TargetScaler.fit(self.batch.targets)
        physical = scaler.to_physical(np.asarray(raw))
        self.assertTrue(np.isfinite(physical).all())


if __name__ == "__main__":
    unittest.main()
