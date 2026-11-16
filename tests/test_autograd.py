import unittest

import numpy as np

from photocat.autograd import Adam, Tensor, embedding_lookup, layernorm


def numerical_grad(function, array, eps=1e-6):
    grad = np.zeros_like(array)
    iterator = np.nditer(array, flags=["multi_index"])
    while not iterator.finished:
        index = iterator.multi_index
        original = array[index]
        array[index] = original + eps
        plus = function()
        array[index] = original - eps
        minus = function()
        array[index] = original
        grad[index] = (plus - minus) / (2 * eps)
        iterator.iternext()
    return grad


class AutogradTest(unittest.TestCase):
    def test_composite_gradients(self):
        rng = np.random.default_rng(0)
        x = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
        weight = Tensor(rng.normal(size=(4, 5)), requires_grad=True)
        bias = Tensor(rng.normal(size=(5,)), requires_grad=True)
        gamma = Tensor(rng.normal(size=(5,)), requires_grad=True)
        beta = Tensor(rng.normal(size=(5,)), requires_grad=True)
        target = rng.normal(size=(3, 5))

        def loss():
            hidden = layernorm(x @ weight + bias, gamma, beta)
            probabilities = hidden.softmax(axis=-1)
            return (probabilities * target).sum() + (hidden * hidden).mean()

        value = loss()
        value.backward()
        for tensor in (x, weight, bias, gamma, beta):
            expected = numerical_grad(lambda: float(loss().data), tensor.data)
            self.assertLess(np.max(np.abs(expected - tensor.grad)), 1e-6)

    def test_embedding_scatter(self):
        rng = np.random.default_rng(1)
        embedding = Tensor(rng.normal(size=(5, 3)), requires_grad=True)
        indices = np.array([[0, 2, 2], [1, 0, 4]])
        weights = rng.normal(size=(2, 3, 3))
        (embedding_lookup(embedding, indices) * Tensor(weights)).sum().backward()
        expected = np.zeros_like(embedding.data)
        for batch in range(2):
            for token in range(3):
                expected[indices[batch, token]] += weights[batch, token]
        np.testing.assert_allclose(embedding.grad, expected)

    def test_adam_reduces_quadratic(self):
        parameter = Tensor([3.0, -2.0], requires_grad=True)
        optimizer = Adam([parameter], lr=0.1)
        for _ in range(200):
            loss = (parameter * parameter).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(float(np.abs(parameter.data).max()), 0.1)


if __name__ == "__main__":
    unittest.main()
