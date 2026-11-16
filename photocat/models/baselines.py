"""Tabular baselines: ridge regression, k-nearest neighbours and a mean predictor.

These provide the "traditional ML" comparison the proposal explicitly mentions.
Descriptors are deliberately strong (composition aggregates, context one-hots,
element fractions) so the comparison is fair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..data.dataset import CONTEXT_FIELDS, Featurizer, _material_descriptor, ph_bucket
from ..data.elements import ELEMENT_IDS, ELEMENT_LIST, MATERIAL_FEATURES
from ..data.synth import Material

def tabular_features(materials: Sequence[Material], featurizer: Featurizer) -> np.ndarray:
    n = len(materials)
    descriptor_width = len(MATERIAL_FEATURES)
    width = (
        descriptor_width
        + 6
        + sum(max(vocab.values(), default=0) + 1 for vocab in featurizer.ctx_vocab.values())
        + len(ELEMENT_LIST)
    )
    matrix = np.zeros((n, width), dtype=np.float64)
    for i, material in enumerate(materials):
        column = 0
        matrix[i, :descriptor_width] = _material_descriptor(material)
        column += descriptor_width
        matrix[i, column : column + 6] = (
            material.ph / 7.0,
            np.log10(material.surface_area + 1.0) / 2.5,
            material.strain * 50.0,
            material.defect,
            {"none": 0.0, "type-II": 1.0, "Z-scheme": 0.5}[material.heterojunction],
            material.dopant_fraction * 10.0,
        )
        column += 6
        for field in CONTEXT_FIELDS:
            vocab = featurizer.ctx_vocab[field]
            size = max(vocab.values(), default=0) + 1
            value = ph_bucket(material.ph) if field == "ph_bucket" else str(getattr(material, field))
            identifier = vocab.get(value, 0)
            matrix[i, column + identifier] = 1.0
            column += size
        total = sum(stoich for _, stoich, _ in material.elements) or 1.0
        for symbol, stoich, _ in material.elements:
            matrix[i, column + ELEMENT_IDS[symbol] - 1] = stoich / total
    return matrix


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Ridge coefficients with an unpenalised intercept."""
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    xc = x - x_mean
    gram = xc.T @ xc + ridge * np.eye(x.shape[1])
    weights = np.linalg.solve(gram, xc.T @ (y - y_mean))
    return np.vstack([weights, y_mean])


def _predict_ridge(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return x @ coefficients[:-1] + coefficients[-1]


@dataclass
class BaselinePredictions:
    eg: np.ndarray
    her: np.ndarray
    stability: np.ndarray


class RidgeBaseline:
    def __init__(self, ridge: float = 1.0) -> None:
        self.ridge = ridge
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.coefficients: np.ndarray | None = None

    def fit(self, materials: Sequence[Material], featurizer: Featurizer) -> "RidgeBaseline":
        x = tabular_features(materials, featurizer)
        targets = np.asarray(
            [[m.eg, np.log1p(m.her), m.stability] for m in materials], dtype=np.float64
        )
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        x_scaled = (x - self.mean) / self.std
        self.coefficients = _fit_ridge(x_scaled, targets, self.ridge)
        return self

    def predict(self, materials: Sequence[Material], featurizer: Featurizer) -> BaselinePredictions:
        if self.coefficients is None:
            raise RuntimeError("baseline is not fitted")
        x = (tabular_features(materials, featurizer) - self.mean) / self.std
        raw = _predict_ridge(x, self.coefficients)
        return BaselinePredictions(
            eg=raw[:, 0],
            her=np.expm1(raw[:, 1]),
            stability=np.clip(raw[:, 2], 0.0, 1.0),
        )


class KNNBaseline:
    def __init__(self, k: int = 12, max_train: int = 4000, seed: int = 0) -> None:
        self.k = k
        self.max_train = max_train
        self.seed = seed
        self.x: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def fit(self, materials: Sequence[Material], featurizer: Featurizer) -> "KNNBaseline":
        x = tabular_features(materials, featurizer)
        targets = np.asarray(
            [[m.eg, np.log1p(m.her), m.stability] for m in materials], dtype=np.float64
        )
        if len(x) > self.max_train:
            rng = np.random.default_rng(self.seed)
            index = rng.choice(len(x), self.max_train, replace=False)
            x, targets = x[index], targets[index]
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        self.x = ((x - self.mean) / self.std).astype(np.float32)
        self.y = targets
        return self

    def predict(self, materials: Sequence[Material], featurizer: Featurizer) -> BaselinePredictions:
        if self.x is None or self.y is None:
            raise RuntimeError("baseline is not fitted")
        x = ((tabular_features(materials, featurizer) - self.mean) / self.std).astype(np.float32)
        predictions = np.zeros((len(x), 3), dtype=np.float64)
        for start in range(0, len(x), 256):
            block = x[start : start + 256]
            block_sq = (block * block).sum(axis=1, keepdims=True)
            distances = (
                block_sq
                - 2.0 * block @ self.x.T
                + (self.x * self.x).sum(axis=1)[None, :]
            )
            nearest = np.argpartition(distances, self.k, axis=1)[:, : self.k]
            for row, neighbours in enumerate(nearest):
                dist = np.maximum(distances[row, neighbours], 0.0)
                weight = 1.0 / (dist + 1e-6)
                predictions[start + row] = (
                    weight[:, None] * self.y[neighbours]
                ).sum(axis=0) / weight.sum()
        return BaselinePredictions(
            eg=predictions[:, 0],
            her=np.expm1(predictions[:, 1]),
            stability=np.clip(predictions[:, 2], 0.0, 1.0),
        )


class MeanBaseline:
    def fit(self, materials: Sequence[Material], featurizer: Featurizer) -> "MeanBaseline":
        targets = np.asarray(
            [[m.eg, np.log1p(m.her), m.stability] for m in materials], dtype=np.float64
        )
        self.mean = targets.mean(axis=0)
        return self

    def predict(self, materials: Sequence[Material], featurizer: Featurizer) -> BaselinePredictions:
        n = len(materials)
        return BaselinePredictions(
            eg=np.full(n, self.mean[0]),
            her=np.full(n, np.expm1(self.mean[1])),
            stability=np.full(n, self.mean[2]),
        )
