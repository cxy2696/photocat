"""Candidate screening: hard physical filters, novelty, Pareto and diversity.

Implements the proposal's candidate-ranking stage:

* hard constraints: ``1.23 <= Eg <= 3.0 eV``, ``HER >= 6``, stable enough;
* novelty: Tanimoto similarity to the training library below 0.8;
* a transparent multi-criteria score (HER, bandgap optimality, stability,
  novelty, ensemble uncertainty, earth abundance);
* greedy diversity selection and a Pareto front for experimental shortlists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from .data.elements import BY_SYMBOL
from .data.fingerprint import (
    DEFAULT_BITS,
    composition_fingerprint,
    max_similarity,
    pairwise_similarity,
)
from .data.synth import Material


@dataclass
class RankConfig:
    eg_min: float = 1.23
    eg_max: float = 3.0
    her_min: float = 6.0
    stability_min: float = 0.55
    novelty_max: float = 0.80
    top_k: int = 50
    diversity_max_sim: float = 0.92
    eg_optimum: float = 2.0
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "her": 0.42,
            "bandgap": 0.18,
            "stability": 0.14,
            "novelty": 0.14,
            "certainty": 0.07,
            "abundance": 0.05,
        }
    )


@dataclass
class Candidate:
    material: Material
    eg: float
    her: float
    stability: float
    eg_std: float
    her_std: float
    similarity: float
    abundance: float
    score: float
    rank: int = 0
    pareto: bool = False

    def to_row(self) -> dict[str, object]:
        material = self.material
        return {
            "rank": self.rank,
            "id": material.id,
            "label": material.label,
            "formula": material.formula,
            "family": material.family,
            "dopant": material.dopant,
            "cocatalyst": material.cocatalyst,
            "method": material.method,
            "ph": material.ph,
            "pred_eg": round(self.eg, 3),
            "eg_std": round(self.eg_std, 3),
            "pred_her": round(self.her, 3),
            "her_std": round(self.her_std, 3),
            "pred_stability": round(self.stability, 3),
            "novelty_similarity": round(self.similarity, 3),
            "abundance_score": round(self.abundance, 3),
            "score": round(self.score, 4),
            "pareto": int(self.pareto),
        }


def _min_abundance(material: Material) -> float:
    values = [BY_SYMBOL[symbol].abundance for symbol, _, role in material.elements if role != "dopant"]
    return min(values) if values else 0.0


def pareto_mask(objectives: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows, all objectives maximised."""
    n = len(objectives)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[i]:
                continue
            if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                mask[i] = False
    return mask


def screen_candidates(
    materials: Sequence[Material],
    predictions: dict[str, np.ndarray],
    library: Sequence[Material],
    config: RankConfig,
    n_bits: int = DEFAULT_BITS,
) -> tuple[list[Candidate], dict[str, float]]:
    """Apply filters, novelty and scoring; return ranked, diverse candidates."""
    eg = np.asarray(predictions["eg"], dtype=np.float64)
    her = np.asarray(predictions["her"], dtype=np.float64)
    stability = np.asarray(predictions["stability"], dtype=np.float64)
    eg_std = np.asarray(predictions.get("eg_std", np.zeros_like(eg)), dtype=np.float64)
    her_std = np.asarray(predictions.get("her_std", np.zeros_like(her)), dtype=np.float64)
    if not (len(materials) == len(eg) == len(her) == len(stability)):
        raise ValueError("predictions and materials must have the same length")

    mask = (
        (eg >= config.eg_min)
        & (eg <= config.eg_max)
        & (her >= config.her_min)
        & (stability >= config.stability_min)
    )
    stage_pass = {"pool": float(len(materials)), "physical": float(mask.sum())}
    if mask.sum() == 0:
        return [], stage_pass

    indices = np.where(mask)[0]
    query = np.stack([composition_fingerprint(materials[i], n_bits) for i in indices])
    library_fingerprints = np.stack([composition_fingerprint(m, n_bits) for m in library])
    similarity, nearest = max_similarity(query, library_fingerprints)
    keep = similarity < config.novelty_max
    stage_pass["novel"] = float(keep.sum())
    if keep.sum() == 0:
        return [], stage_pass

    indices = indices[keep]
    similarity = similarity[keep]
    nearest = nearest[keep]
    eg = eg[indices]
    her = her[indices]
    stability = stability[indices]
    eg_std = eg_std[indices]
    her_std = her_std[indices]

    log_her = np.log1p(her)
    her_mean = log_her.mean()
    her_scale = log_her.std() or 1.0
    her_score = (log_her - her_mean) / her_scale
    bandgap_score = np.exp(-0.5 * ((eg - config.eg_optimum) / 0.55) ** 2)
    novelty_score = np.clip((config.novelty_max - similarity) / config.novelty_max, 0.0, 1.0)
    certainty = np.exp(-(eg_std / 0.30 + her_std / 2.0))
    abundance = np.clip(
        np.log10(np.asarray([_min_abundance(materials[i]) for i in indices]) + 1.0) / 3.0,
        0.0,
        1.0,
    )
    weights = config.weights
    score = (
        weights["her"] * her_score
        + weights["bandgap"] * bandgap_score
        + weights["stability"] * stability
        + weights["novelty"] * novelty_score
        + weights["certainty"] * certainty
        + weights["abundance"] * abundance
    )

    objectives = np.stack([her, stability, novelty_score, -np.abs(eg - config.eg_optimum)], axis=1)
    pareto = pareto_mask(objectives)

    # Greedy score-then-diversity selection, with unique material labels so the
    # shortlist does not spend slots on the same composition under different
    # synthesis conditions.
    selected: list[int] = []
    seen_labels: set[str] = set()
    for local in np.argsort(-score):
        if len(selected) >= config.top_k:
            break
        label = materials[int(indices[local])].label
        if label in seen_labels:
            continue
        if any(
            pairwise_similarity(query[local], query[other]) >= config.diversity_max_sim
            for other in selected
        ):
            continue
        selected.append(int(local))
        seen_labels.add(label)
    if len(selected) < config.top_k:
        for local in np.argsort(-score):
            if len(selected) >= config.top_k:
                break
            if int(local) not in selected:
                selected.append(int(local))

    candidates: list[Candidate] = []
    for rank, local in enumerate(selected, start=1):
        global_index = int(indices[local])
        candidate = Candidate(
            material=materials[global_index],
            eg=float(eg[local]),
            her=float(her[local]),
            stability=float(stability[local]),
            eg_std=float(eg_std[local]),
            her_std=float(her_std[local]),
            similarity=float(similarity[local]),
            abundance=float(abundance[local]),
            score=float(score[local]),
            rank=rank,
            pareto=bool(pareto[local]),
        )
        candidates.append(candidate)
    stage_pass["selected"] = float(len(candidates))
    stage_pass["pareto"] = float(pareto.sum())
    stage_pass["similarity_median"] = float(np.median(similarity[selected])) if selected else float("nan")
    stage_pass["similarity_mean"] = float(np.mean(similarity[selected])) if selected else float("nan")
    return candidates, stage_pass


def novelty_of(candidate: Candidate, library: Sequence[Material], n_bits: int = DEFAULT_BITS) -> float:
    """Recompute similarity for one candidate (used by tests/explainers)."""
    query = composition_fingerprint(candidate.material, n_bits)[None, :]
    library_fingerprints = np.stack([composition_fingerprint(m, n_bits) for m in library])
    return float(max_similarity(query, library_fingerprints)[0][0])


def config_dict(config: RankConfig) -> dict[str, object]:
    return asdict(config)
