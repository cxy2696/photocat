"""RBF kernel ridge regression for small process datasets.

The EES photocatalytic HER table has 552 complete records and 34 papers.  A
deep composition model overfits this regime; a regularised kernel model is the
appropriate baseline and gives a stronger paper-held-out estimate.  The
implementation is NumPy-only and deliberately small enough to audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class KernelRidge:
    gamma: float = 0.03
    alpha: float = 0.1

    def __post_init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.x_train_: np.ndarray | None = None
        self.coefficients_: np.ndarray | None = None
        self.y_mean_: float = 0.0

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.scale_

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelRidge":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-9] = 1.0
        self.x_train_ = self._standardise(x)
        self.y_mean_ = float(y.mean())
        squared = (
            (self.x_train_[:, None, :] - self.x_train_[None, :, :]) ** 2
        ).sum(axis=-1)
        kernel = np.exp(-self.gamma * squared)
        system = kernel + self.alpha * np.eye(len(x))
        self.coefficients_ = np.linalg.solve(system, y - self.y_mean_)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_train_ is None or self.coefficients_ is None:
            raise RuntimeError("kernel ridge is not fitted")
        x = self._standardise(np.asarray(x, dtype=np.float64))
        squared = ((x[:, None, :] - self.x_train_[None, :, :]) ** 2).sum(axis=-1)
        return np.exp(-self.gamma * squared) @ self.coefficients_ + self.y_mean_


def group_cross_validate(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[object],
    folds: int = 5,
    seed: int = 0,
    gamma: float = 0.03,
    alpha: float = 0.1,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Leave-papers-out cross-validation; returns aggregated predictions."""
    groups_array = np.asarray(groups, dtype=object)
    unique = np.asarray(sorted(set(groups_array.tolist())), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    assignment = np.array_split(unique, folds)
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    summaries: list[dict[str, object]] = []
    for fold_index, held_out in enumerate(assignment):
        test_mask = np.isin(groups_array, held_out)
        train_mask = ~test_mask
        model = KernelRidge(gamma=gamma, alpha=alpha).fit(x[train_mask], y[train_mask])
        predictions[test_mask] = model.predict(x[test_mask])
        summaries.append(
            {
                "fold": fold_index,
                "papers": int(len(held_out)),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
            }
        )
    return predictions, summaries
