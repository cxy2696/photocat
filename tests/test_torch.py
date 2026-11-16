import unittest
from pathlib import Path

import numpy as np

from photocat.data.dataset import Featurizer
from photocat.data.elements import ELEMENT_IDS
from photocat.data.synth import generate_dataset
from photocat.models.torch_transformer import (
    TORCH_AVAILABLE,
    TorchGraphAugmentedTransformer,
)
from photocat.models.transformer import GraphAugmentedTransformer

if TORCH_AVAILABLE:
    import torch
    from photocat.training_torch import TorchTrainer, batch_to_torch
    from photocat.training import TrainConfig


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch optional dependency is not installed")
class TorchParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = generate_dataset(n=300, seed=11)
        cls.featurizer = Featurizer.fit(cls.bundle.train)
        cls.batch = cls.featurizer.transform(cls.bundle.test[:24])
        cls.ctx_sizes = [cls.featurizer.vocab_size(field) for field in cls.featurizer.ctx_vocab]
        cls.vocab = max(ELEMENT_IDS.values()) + 1
        cls.numpy_model = GraphAugmentedTransformer(
            cls.vocab,
            cls.ctx_sizes,
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=48,
            gnn_dim=16,
            dropout=0.0,
            seed=3,
        )
        cls.torch_model = TorchGraphAugmentedTransformer(
            cls.vocab,
            cls.ctx_sizes,
            d_model=32,
            n_heads=4,
            n_layers=2,
            d_ff=48,
            gnn_dim=16,
            dropout=0.0,
            dtype=torch.float64,
            seed=3,
        ).load_from_numpy(cls.numpy_model)
        cls.torch_model.eval()

    def test_forward_matches_numpy(self):
        inputs = batch_to_torch(self.batch, dtype=torch.float64)
        with torch.no_grad():
            torch_out = self.torch_model(inputs, return_attention=True)
        numpy_out = self.numpy_model.forward(self.batch, return_attention=True)
        for name, torch_value, numpy_value in (
            ("eg", torch_out.eg, numpy_out.eg),
            ("her", torch_out.her, numpy_out.her),
            ("stability", torch_out.stability, numpy_out.stability),
        ):
            np.testing.assert_allclose(
                torch_value.cpu().numpy(),
                numpy_value.data,
                rtol=1e-9,
                atol=1e-9,
                err_msg=name,
            )
        self.assertEqual(len(torch_out.attention), len(numpy_out.attention))
        for torch_attn, numpy_attn in zip(torch_out.attention, numpy_out.attention):
            np.testing.assert_allclose(
                torch_attn.cpu().numpy(),
                numpy_attn,
                rtol=1e-9,
                atol=1e-9,
            )

    def test_attention_is_normalised(self):
        inputs = batch_to_torch(self.batch, dtype=torch.float64)
        with torch.no_grad():
            output = self.torch_model(inputs, return_attention=True)
        attention = output.attention[0]
        self.assertEqual(tuple(attention.shape), (24, 4, 15, 15))
        np.testing.assert_allclose(
            attention.sum(dim=-1).cpu().numpy(), 1.0, atol=1e-9
        )
        key_mask = torch.as_tensor(
            self.batch.token_mask[:, None, None, :], dtype=torch.bool
        )
        leak = torch.where(key_mask, torch.zeros_like(attention), attention)
        self.assertLess(float(leak.max()), 1e-8)

    def test_torch_training_smoke(self):
        model = TorchGraphAugmentedTransformer(
            self.vocab,
            self.ctx_sizes,
            d_model=32,
            n_heads=4,
            n_layers=1,
            d_ff=48,
            gnn_dim=16,
            dropout=0.1,
            dtype=torch.float32,
            seed=4,
        )
        trainer = TorchTrainer(TrainConfig(epochs=1, batch_size=64, verbose=False))
        result = trainer.train(model, self.featurizer, self.bundle.train[:180], self.bundle.val)
        self.assertEqual(len(result.history.epochs), 1)
        self.assertTrue(np.isfinite(result.history.best_val_loss))

    def test_torch_save_load(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            self.torch_model.save(str(path))
            restored = TorchGraphAugmentedTransformer.load(str(path), dtype=torch.float64)
            restored.eval()
            inputs = batch_to_torch(self.batch, dtype=torch.float64)
            with torch.no_grad():
                original = self.torch_model(inputs).eg
                copied = restored(inputs).eg
            np.testing.assert_allclose(
                original.cpu().numpy(), copied.cpu().numpy(), rtol=1e-9, atol=1e-9
            )


if __name__ == "__main__":
    unittest.main()
