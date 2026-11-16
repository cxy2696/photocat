"""Deep-ensemble inference with epistemic uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..data.dataset import Featurizer
from ..data.synth import Material
from ..training import TrainingResult, forward_batches


@dataclass
class EnsemblePrediction:
    eg: np.ndarray
    her: np.ndarray
    stability: np.ndarray
    eg_std: np.ndarray
    her_std: np.ndarray
    stability_std: np.ndarray
    member_eg: np.ndarray
    member_her: np.ndarray
    member_stability: np.ndarray

    def __len__(self) -> int:
        return len(self.eg)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "eg": self.eg,
            "her": self.her,
            "stability": self.stability,
            "eg_std": self.eg_std,
            "her_std": self.her_std,
            "stability_std": self.stability_std,
        }


def ensemble_predict(
    results: Sequence[TrainingResult],
    featurizer: Featurizer,
    materials: Sequence[Material],
    batch_size: int = 256,
) -> EnsemblePrediction:
    if not results:
        raise ValueError("no trained models supplied")
    batch = featurizer.transform(materials)
    members: list[np.ndarray] = []
    for result in results:
        raw = np.asarray(forward_batches(result.model, batch, batch_size=batch_size)["raw"])
        members.append(result.scaler.to_physical(raw))
    stacked = np.stack(members, axis=0)  # (K, B, 3)
    return EnsemblePrediction(
        eg=stacked[:, :, 0].mean(axis=0),
        her=stacked[:, :, 1].mean(axis=0),
        stability=stacked[:, :, 2].mean(axis=0),
        eg_std=stacked[:, :, 0].std(axis=0),
        her_std=stacked[:, :, 1].std(axis=0),
        stability_std=stacked[:, :, 2].std(axis=0),
        member_eg=stacked[:, :, 0],
        member_her=stacked[:, :, 1],
        member_stability=stacked[:, :, 2],
    )
