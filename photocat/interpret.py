"""Interpretability: attention rollout, element Shapley values and ablations.

The proposal calls for SHAP-style feature importance and attention
visualisation.  Because the model consumes discrete element tokens and context
tokens, we compute *exact* Shapley values over element coalitions (at most six
elements, so at most 64 model calls per material) and a standard attention
rollout from the ``[CLS]`` token.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from .data.dataset import CONTEXT_FIELDS, Featurizer
from .data.elements import MATERIAL_FEATURES
from .data.synth import Material, compute_descriptors
from .models.ensemble import ensemble_predict
from .training import TrainingResult

EMPTY_DESCRIPTORS = {
    **{name: 0.0 for name in MATERIAL_FEATURES},
    "ph": 7.0,
    "log_surface_area": math.log10(26.0),
    "strain": 0.0,
    "defect": 0.0,
}


@dataclass
class AttentionExplanation:
    tokens: list[str]
    importance: np.ndarray


def token_labels(material: Material) -> list[str]:
    labels = ["[CLS]"]
    total = sum(stoich for _, stoich, _ in material.elements) or 1.0
    for symbol, stoich, role in material.elements:
        labels.append(f"{symbol} ({stoich / total:.2f}, {role})")
    for field in CONTEXT_FIELDS:
        if field == "ph_bucket":
            from .data.dataset import ph_bucket

            value = ph_bucket(material.ph)
        else:
            value = getattr(material, field)
        labels.append(f"{field}={value}")
    return labels


def attention_rollout(
    attention: Sequence[np.ndarray],
    material: Material,
    batch_index: int = 0,
    head_reduce: str = "mean",
) -> AttentionExplanation:
    """Propagate attention from tokens to CLS across layers."""
    token_count = attention[0].shape[-1]
    rollout = np.eye(token_count)
    for layer in attention:
        layer_attention = layer[batch_index]
        if head_reduce == "mean":
            layer_attention = layer_attention.mean(axis=0)
        else:
            layer_attention = layer_attention.max(axis=0)
        layer_attention = 0.5 * layer_attention + 0.5 * np.eye(token_count)
        layer_attention = layer_attention / np.maximum(
            layer_attention.sum(axis=-1, keepdims=True), 1e-9
        )
        rollout = layer_attention @ rollout
    labels = token_labels(material)
    # Pad labels if the batch has more token slots than this material uses.
    while len(labels) < token_count:
        labels.append("[PAD]")
    return AttentionExplanation(tokens=labels[:token_count], importance=rollout[0, :token_count])


def _subset_material(material: Material, subset: Sequence[tuple[str, float, str]]) -> Material:
    clone = copy.deepcopy(material)
    clone.elements = tuple(subset)
    if material.dopant != "none" and material.dopant not in {symbol for symbol, _, _ in subset}:
        clone.dopant = "none"
        clone.dopant_fraction = 0.0
    if subset:
        clone.descriptors = compute_descriptors(
            subset,
            material.family,
            material.ph,
            material.surface_area,
            material.strain,
            material.defect,
        )
    else:
        clone.descriptors = dict(EMPTY_DESCRIPTORS)
    return clone


def element_shapley(
    results: Sequence[TrainingResult],
    featurizer: Featurizer,
    material: Material,
    target: str = "her",
) -> dict[str, object]:
    """Exact Shapley values over element coalitions for one prediction."""
    elements = list(material.elements)
    n = len(elements)
    coalitions: list[Material] = []
    for mask in range(1 << n):
        subset = [elements[i] for i in range(n) if mask & (1 << i)]
        coalitions.append(_subset_material(material, subset))
    prediction = ensemble_predict(results, featurizer, coalitions)
    values = prediction.her if target == "her" else prediction.eg
    factorials = [math.factorial(i) for i in range(n + 1)]
    attribution: dict[str, float] = {}
    for i, (symbol, _, _) in enumerate(elements):
        phi = 0.0
        others = [j for j in range(n) if j != i]
        for size in range(n):
            weight = factorials[size] * factorials[n - size - 1] / factorials[n]
            for subset in combinations(others, size):
                mask_without = sum(1 << j for j in subset)
                mask_with = mask_without | (1 << i)
                phi += weight * (values[mask_with] - values[mask_without])
        attribution[symbol] = float(phi)
    total = float(prediction.her[0] if target == "her" else prediction.eg[0])
    return {
        "material_id": material.id,
        "label": material.label,
        "target": target,
        "prediction": total,
        "attribution": attribution,
    }


def permutation_importance(
    results: Sequence[TrainingResult],
    featurizer: Featurizer,
    materials: Sequence[Material],
    fields: Sequence[str] = ("method", "cocatalyst", "ph", "surface_area", "heterojunction"),
    repeats: int = 5,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Shuffle one condition at a time and measure error degradation."""
    rng = np.random.default_rng(seed)
    base = ensemble_predict(results, featurizer, materials)
    base_her_rmse = float(
        np.sqrt(
            np.mean(
                (np.log1p(np.array([m.her for m in materials])) - np.log1p(np.clip(base.her, 0.0, None))) ** 2
            )
        )
    )
    importances: list[dict[str, object]] = []
    for field in fields:
        scores: list[float] = []
        for _ in range(repeats):
            clones = [copy.deepcopy(material) for material in materials]
            values = [getattr(material, field) for material in clones]
            shuffled = list(rng.permutation(values))
            for clone, value in zip(clones, shuffled):
                setattr(clone, field, value.item() if hasattr(value, "item") else value)
                if field in ("ph", "surface_area"):
                    clone.descriptors = compute_descriptors(
                        clone.elements,
                        clone.family,
                        float(clone.ph),
                        float(clone.surface_area),
                        clone.strain,
                        clone.defect,
                    )
            prediction = ensemble_predict(results, featurizer, clones)
            scores.append(
                float(
                    np.sqrt(
                        np.mean(
                            (np.log1p(np.array([m.her for m in clones])) - np.log1p(np.clip(prediction.her, 0.0, None))) ** 2
                        )
                    )
                )
            )
        importances.append(
            {
                "field": field,
                "delta_rmse": float(np.mean(scores) - base_her_rmse),
                "rmse_shuffled": float(np.mean(scores)),
                "base_rmse": base_her_rmse,
            }
        )
    importances.sort(key=lambda row: float(row["delta_rmse"]), reverse=True)
    return importances
