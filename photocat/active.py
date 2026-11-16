"""Simulated active-learning loop for experimental iteration savings.

The proposal claims a 50% reduction in experimental iterations.  We test that
claim in silico: a cheap ridge surrogate is retrained after each batch of
"experiments" (labels revealed), with bootstrap uncertainty and a bandgap-window
acquisition.  Recall of the true top-performing pool materials is compared with
random sampling at the same label budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .data.dataset import Featurizer
from .data.synth import Material
from .models.baselines import _fit_ridge, _predict_ridge, tabular_features


@dataclass
class ActiveLearningConfig:
    initial_labels: int = 150
    batch_size: int = 100
    rounds: int = 12
    n_bootstrap: int = 5
    beta: float = 0.35
    seed: int = 0
    target_quantile: float = 0.99
    eg_min: float = 1.23
    eg_max: float = 3.0
    stability_min: float = 0.55


@dataclass
class ActiveLearningResult:
    curve: list[dict[str, float]] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)


def _target_indices(pool: Sequence[Material], config: ActiveLearningConfig) -> np.ndarray:
    eg = np.asarray([material.eg for material in pool])
    her = np.asarray([material.her for material in pool])
    stability = np.asarray([material.stability for material in pool])
    eligible = (eg >= config.eg_min) & (eg <= config.eg_max) & (stability >= config.stability_min)
    eligible_indices = np.where(eligible)[0]
    if len(eligible_indices) == 0:
        return np.array([], dtype=np.int64)
    threshold = np.quantile(her[eligible_indices], config.target_quantile)
    targets = eligible_indices[her[eligible_indices] >= threshold]
    if len(targets) < 10:
        targets = eligible_indices[np.argsort(-her[eligible_indices])[: max(10, len(targets))]]
    return np.sort(targets)


def _fit_and_predict(
    x_labeled: np.ndarray,
    y_labeled: np.ndarray,
    x_pool: np.ndarray,
    ridge: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_labeled.mean(axis=0)
    std = x_labeled.std(axis=0)
    std[std < 1e-8] = 1.0
    xl = (x_labeled - mean) / std
    xp = (x_pool - mean) / std
    coefficients = _fit_ridge(xl, y_labeled, ridge)
    raw = _predict_ridge(xp, coefficients)
    return raw[:, 0], raw[:, 1], raw[:, 2]


def simulate_active_learning(
    train_materials: Sequence[Material],
    pool_materials: Sequence[Material],
    featurizer: Featurizer,
    config: ActiveLearningConfig | None = None,
) -> ActiveLearningResult:
    config = config or ActiveLearningConfig()
    rng = np.random.default_rng(config.seed)
    x_train = tabular_features(train_materials, featurizer)
    y_train = np.asarray(
        [[m.eg, np.log1p(m.her), m.stability] for m in train_materials], dtype=np.float64
    )
    x_pool = tabular_features(pool_materials, featurizer)
    y_pool = np.asarray(
        [[m.eg, np.log1p(m.her), m.stability] for m in pool_materials], dtype=np.float64
    )
    targets = _target_indices(pool_materials, config)
    if len(targets) == 0:
        raise RuntimeError("no eligible target materials in the active-learning pool")

    initial = min(config.initial_labels, len(pool_materials))
    initial_index = rng.choice(len(pool_materials), initial, replace=False)
    labeled = np.zeros(len(pool_materials), dtype=bool)
    labeled[initial_index] = True

    active_curve: list[tuple[int, float]] = []
    random_curve: list[tuple[int, float]] = []
    active_selected: set[int] = set(initial_index.tolist())
    random_selected: set[int] = set(initial_index.tolist())

    def revealed(selection: set[int]) -> np.ndarray:
        indices = np.array(sorted(selection), dtype=np.int64)
        return np.concatenate(
            [
                y_train,
                y_pool[indices],
            ],
            axis=0,
        )

    def recall(selection: set[int]) -> float:
        found = len(set(targets.tolist()) & selection)
        return found / len(targets)

    for round_index in range(config.rounds + 1):
        budget = initial + round_index * config.batch_size
        active_curve.append((budget, recall(active_selected)))
        random_curve.append((budget, recall(random_selected)))
        if round_index == config.rounds or len(active_selected) >= len(pool_materials):
            break

        # Active: bootstrap ridge ensemble over all revealed labels.
        x_labeled = np.concatenate([x_train, x_pool[np.where(labeled)[0]]], axis=0)
        y_labeled = revealed(set(np.where(labeled)[0].tolist()))
        predictions = []
        for _ in range(config.n_bootstrap):
            sample = rng.integers(0, len(x_labeled), len(x_labeled))
            eg, logher, _ = _fit_and_predict(x_labeled[sample], y_labeled[sample], x_pool)
            predictions.append(np.stack([eg, logher], axis=1))
        prediction = np.stack(predictions)  # (B, P, 2)
        mean_pred = prediction.mean(axis=0)
        std_pred = prediction.std(axis=0)
        acquisition = (
            mean_pred[:, 1]
            + config.beta * std_pred[:, 1]
            - 0.5 * np.maximum(0.0, np.abs(mean_pred[:, 0] - 2.0) - 0.8)
        )
        eligible = (
            (mean_pred[:, 0] >= config.eg_min)
            & (mean_pred[:, 0] <= config.eg_max)
            & ~labeled
        )
        pool = np.where(eligible)[0]
        if len(pool) == 0:
            pool = np.where(~labeled)[0]
        chosen = pool[np.argsort(-acquisition[pool])[: config.batch_size]]
        active_selected.update(int(i) for i in chosen)
        labeled[chosen] = True

        # Random baseline at the same budget.
        remaining = np.where(~np.isin(np.arange(len(pool_materials)), list(random_selected)))[0]
        if len(remaining) > 0:
            random_chosen = rng.choice(remaining, min(config.batch_size, len(remaining)), replace=False)
            random_selected.update(int(i) for i in random_chosen)

    def labels_to_recall(curve: list[tuple[int, float]], level: float) -> float:
        for budget, value in curve:
            if value >= level:
                return float(budget)
        return float("nan")

    def reduction(level: float) -> float:
        active_budget = labels_to_recall(active_curve, level)
        random_budget = labels_to_recall(random_curve, level)
        if not np.isfinite(active_budget) or not np.isfinite(random_budget) or random_budget <= 0:
            return float("nan")
        return 1.0 - active_budget / random_budget

    def reduction_lower(level: float) -> float:
        """Conservative savings when random never reaches ``level`` in budget."""
        active_budget = labels_to_recall(active_curve, level)
        if not np.isfinite(active_budget):
            return float("nan")
        random_budget = labels_to_recall(random_curve, level)
        if np.isfinite(random_budget) and random_budget > 0:
            return 1.0 - active_budget / random_budget
        tested_budget = random_curve[-1][0]
        return 1.0 - active_budget / tested_budget if tested_budget > 0 else float("nan")

    result = ActiveLearningResult()
    result.curve = [
        {
            "budget": float(active_budget),
            "active_recall": float(active_value),
            "random_recall": float(random_value),
        }
        for (active_budget, active_value), (_, random_value) in zip(active_curve, random_curve)
    ]
    result.summary = {
        "n_targets": float(len(targets)),
        "n_pool": float(len(pool_materials)),
        "initial_labels": float(initial),
        "final_active_recall": float(active_curve[-1][1]),
        "final_random_recall": float(random_curve[-1][1]),
        "labels_to_50pct_active": labels_to_recall(active_curve, 0.5),
        "labels_to_50pct_random": labels_to_recall(random_curve, 0.5),
        "reduction_at_50pct": reduction(0.5),
        "reduction_at_50pct_lower_bound": reduction_lower(0.5),
        "labels_to_80pct_active": labels_to_recall(active_curve, 0.8),
        "labels_to_80pct_random": labels_to_recall(random_curve, 0.8),
        "reduction_at_80pct": reduction(0.8),
        "reduction_at_80pct_lower_bound": reduction_lower(0.8),
        "recall_at_300_active": _recall_at(active_curve, 300),
        "recall_at_300_random": _recall_at(random_curve, 300),
    }
    return result


def _recall_at(curve: list[tuple[int, float]], budget: float) -> float:
    """Recall at or before a new-label budget."""
    value = 0.0
    for curve_budget, curve_value in curve:
        if curve_budget <= budget:
            value = curve_value
    return float(value)
