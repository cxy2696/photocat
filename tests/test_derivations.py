import unittest

import numpy as np

from photocat.autograd import Tensor, mse_loss
from photocat.metrics import conformal_quantile
from photocat.models.kernel import KernelRidge
from photocat.training import TargetScaler


class DerivationTest(unittest.TestCase):
    def test_masked_mse_matches_manual_expression(self):
        prediction = np.array([1.0, 2.0, 3.0, 4.0])
        target = np.array([1.5, 2.5, 99.0, 4.5])
        weight = np.array([1.0, 2.0, 0.0, 1.0])
        tensor = Tensor(prediction, requires_grad=True)
        loss = mse_loss(tensor, target, weight)
        expected = np.sum(weight * (prediction - target) ** 2) / np.sum(weight)
        self.assertAlmostEqual(float(loss.data), expected, places=12)
        loss.backward()
        expected_grad = 2.0 * weight * (prediction - target) / np.sum(weight)
        np.testing.assert_allclose(tensor.grad, expected_grad, atol=1e-12)

    def test_kernel_ridge_matches_closed_form(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(12, 3))
        y = x[:, 0] - 0.5 * x[:, 1] + 0.1 * rng.normal(size=12)
        gamma, alpha = 0.2, 0.3
        model = KernelRidge(gamma=gamma, alpha=alpha).fit(x, y)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-9] = 1.0
        xs = (x - mean) / scale
        kernel = np.exp(-gamma * ((xs[:, None, :] - xs[None, :, :]) ** 2).sum(-1))
        coefficients = np.linalg.solve(kernel + alpha * np.eye(len(x)), y - y.mean())
        expected = kernel @ coefficients + y.mean()
        np.testing.assert_allclose(model.predict(x), expected, atol=1e-10)

    def test_conformal_quantile_uses_finite_sample_correction(self):
        residuals = np.arange(1.0, 101.0)
        quantile = conformal_quantile(residuals, coverage=0.9)
        # ceil((100 + 1) * 0.9) / 100 = 0.91; the "higher" order statistic is 92.
        self.assertEqual(quantile, 92.0)

    def test_target_scaler_masks_unobserved_targets(self):
        targets = np.array(
            [
                [1.0, np.nan, np.nan],
                [3.0, np.nan, np.nan],
                [5.0, np.nan, np.nan],
            ]
        )
        scaler = TargetScaler.fit(targets)
        self.assertTrue(scaler.observed[0])
        self.assertFalse(scaler.observed[1])
        self.assertFalse(scaler.observed[2])
        physical = scaler.to_physical(np.zeros((2, 3)))
        self.assertTrue(np.isfinite(physical[:, 0]).all())
        self.assertTrue(np.isnan(physical[:, 1:]).all())


if __name__ == "__main__":
    unittest.main()
