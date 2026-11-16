"""Featurisation of materials into graph tokens for the GNN + transformer.

Every material becomes

* a small knowledge-graph subgraph: one material node plus up to six element
  nodes (``hasElement`` edges and element co-occurrence edges), and
* a token sequence: ``[CLS]`` + element tokens + seven context tokens
  (family, crystal system, co-catalyst, method, dopant, pH bucket,
  heterojunction).

Element tokens carry their GNN node embedding, and the attention scores receive
a graph-derived bias from the same adjacency matrix.  Context vocabularies are
fitted on the training split only; unseen categories map to id 0 so that
held-out families can still be scored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .elements import (
    ELEMENT_FEATURES,
    ELEMENT_IDS,
    MATERIAL_FEATURES,
    element_feature_matrix,
    standardise_element_features,
)
from .synth import Material

CONTEXT_FIELDS: tuple[str, ...] = (
    "family",
    "crystal_system",
    "cocatalyst",
    "method",
    "dopant",
    "ph_bucket",
    "heterojunction",
    "sacrificial_agent",
)

ROLE_IDS: dict[str, int] = {"cation": 1, "anion": 2, "dopant": 3, "framework": 4}
ROLE_TOKEN_TYPES: dict[str, int] = {"cation": 2, "anion": 3, "dopant": 4, "framework": 5}

MAX_ELEMENTS = 6
MAX_TOKENS = 1 + MAX_ELEMENTS + len(CONTEXT_FIELDS)

NUMERIC_FEATURES: tuple[str, ...] = (
    "ph",
    "log_surface_area",
    "strain",
    "defect",
    "heterojunction",
    "dopant_fraction",
    "cocatalyst_loading",
    "light_intensity",
    "catalyst_loading",
    "sacrificial_concentration",
)
NODE_FEATURES = len(ELEMENT_FEATURES) + 1 + len(ROLE_IDS)  # + is_material + role one-hot


def ph_bucket(ph: float) -> str:
    return str(int(np.clip((ph - 1.0) // 2.0, 0, 5)))


def _context_value(material: Material, field: str) -> str:
    if field == "ph_bucket":
        return ph_bucket(material.ph)
    return str(getattr(material, field))


def _heterojunction_flag(kind: str) -> float:
    return {"none": 0.0, "type-II": 1.0, "Z-scheme": 0.5}[kind]


@dataclass
class FeatureBatch:
    """All tensors needed for a forward pass; slicing keeps tensors aligned."""

    token_type: np.ndarray  # (B, T) int
    element_id: np.ndarray  # (B, T) int
    role_id: np.ndarray  # (B, T) int
    stoich_feat: np.ndarray  # (B, T, 2) float
    ctx_type: np.ndarray  # (B, T) int
    ctx_value: np.ndarray  # (B, T) int
    token_mask: np.ndarray  # (B, T) bool
    token_node: np.ndarray  # (B, T) int
    node_select: np.ndarray  # (B, T, N) float
    node_features: np.ndarray  # (B, N, F) float
    node_mask: np.ndarray  # (B, N) bool
    adjacency: np.ndarray  # (B, N, N) float
    pair_bias: np.ndarray  # (B, T, T) float
    numeric: np.ndarray  # (B, 10) float
    targets: np.ndarray  # (B, 3) float: eg, log1p(her), stability
    ids: list[str]
    labels: list[str]

    def __len__(self) -> int:
        return self.token_type.shape[0]

    def take(self, indices: np.ndarray | Sequence[int]) -> "FeatureBatch":
        index = np.asarray(indices)
        return FeatureBatch(
            token_type=self.token_type[index],
            element_id=self.element_id[index],
            role_id=self.role_id[index],
            stoich_feat=self.stoich_feat[index],
            ctx_type=self.ctx_type[index],
            ctx_value=self.ctx_value[index],
            token_mask=self.token_mask[index],
            token_node=self.token_node[index],
            node_select=self.node_select[index],
            node_features=self.node_features[index],
            node_mask=self.node_mask[index],
            adjacency=self.adjacency[index],
            pair_bias=self.pair_bias[index],
            numeric=self.numeric[index],
            targets=self.targets[index],
            ids=[self.ids[i] for i in index],
            labels=[self.labels[i] for i in index],
        )


class Featurizer:
    """Fitted preprocessing state (context vocab + normalisation)."""

    def __init__(
        self,
        ctx_vocab: dict[str, dict[str, int]],
        material_mean: np.ndarray,
        material_std: np.ndarray,
        max_elements: int = MAX_ELEMENTS,
    ) -> None:
        self.ctx_vocab = ctx_vocab
        self.material_mean = np.asarray(material_mean, dtype=np.float64)
        self.material_std = np.asarray(material_std, dtype=np.float64)
        self.material_std[self.material_std < 1e-8] = 1.0
        self.max_elements = max_elements
        self.element_mean, self.element_std = standardise_element_features()

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, materials: Sequence[Material], max_elements: int = MAX_ELEMENTS) -> "Featurizer":
        if not materials:
            raise ValueError("cannot fit a featurizer on an empty list")
        ctx_vocab: dict[str, dict[str, int]] = {}
        for field in CONTEXT_FIELDS:
            values = sorted({_context_value(material, field) for material in materials})
            ctx_vocab[field] = {value: i + 1 for i, value in enumerate(values)}
        matrix = np.asarray([_material_descriptor(material) for material in materials])
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(ctx_vocab, mean, std, max_elements=max_elements)

    # ----------------------------------------------------------------- transform
    def transform(self, materials: Sequence[Material]) -> FeatureBatch:
        batch_size = len(materials)
        num_nodes = 1 + self.max_elements
        token_type = np.zeros((batch_size, MAX_TOKENS), dtype=np.int64)
        element_id = np.zeros((batch_size, MAX_TOKENS), dtype=np.int64)
        role_id = np.zeros((batch_size, MAX_TOKENS), dtype=np.int64)
        stoich_feat = np.zeros((batch_size, MAX_TOKENS, 2), dtype=np.float64)
        ctx_type = np.zeros((batch_size, MAX_TOKENS), dtype=np.int64)
        ctx_value = np.zeros((batch_size, MAX_TOKENS), dtype=np.int64)
        token_mask = np.zeros((batch_size, MAX_TOKENS), dtype=bool)
        token_node = np.zeros((batch_size, MAX_TOKENS), dtype=np.int64)
        node_features = np.zeros((batch_size, num_nodes, NODE_FEATURES), dtype=np.float64)
        node_mask = np.zeros((batch_size, num_nodes), dtype=bool)
        adjacency = np.zeros((batch_size, num_nodes, num_nodes), dtype=np.float64)
        numeric = np.zeros((batch_size, len(NUMERIC_FEATURES)), dtype=np.float64)
        targets = np.full((batch_size, 3), np.nan, dtype=np.float64)
        ids: list[str] = []
        labels: list[str] = []

        for b, material in enumerate(materials):
            total_stoich = sum(stoich for _, stoich, _ in material.elements) or 1.0
            position = 0
            # CLS token, anchored on the material node.
            token_type[b, position] = 1
            token_mask[b, position] = True
            token_node[b, position] = 0
            node_mask[b, 0] = True
            node_features[b, 0, : len(MATERIAL_FEATURES)] = (
                _material_descriptor(material) - self.material_mean
            ) / self.material_std
            node_features[b, 0, len(ELEMENT_FEATURES)] = 1.0  # is_material
            position += 1

            for j, (symbol, stoich, role) in enumerate(material.elements[: self.max_elements]):
                node_index = 1 + j
                token_type[b, position] = ROLE_TOKEN_TYPES[role]
                element_id[b, position] = ELEMENT_IDS[symbol]
                role_id[b, position] = ROLE_IDS[role]
                fraction = stoich / total_stoich
                stoich_feat[b, position] = (fraction, np.log1p(stoich) / 2.0)
                token_mask[b, position] = True
                token_node[b, position] = node_index
                node_mask[b, node_index] = True
                element_row = ELEMENT_IDS[symbol]
                node_features[b, node_index, : len(ELEMENT_FEATURES)] = (
                    _raw_element_row(element_row) - self.element_mean
                ) / self.element_std
                node_features[b, node_index, len(ELEMENT_FEATURES) + ROLE_IDS[role]] = 1.0
                adjacency[b, 0, node_index] = fraction
                adjacency[b, node_index, 0] = fraction
                position += 1

            # Element-element co-occurrence edges.
            for j in range(min(len(material.elements), self.max_elements)):
                for k in range(j + 1, min(len(material.elements), self.max_elements)):
                    frac_j = material.elements[j][1] / total_stoich
                    frac_k = material.elements[k][1] / total_stoich
                    weight = 2.0 * frac_j * frac_k
                    adjacency[b, 1 + j, 1 + k] += weight
                    adjacency[b, 1 + k, 1 + j] += weight

            # Context tokens.
            for field in CONTEXT_FIELDS:
                value = _context_value(material, field)
                token_type[b, position] = 6
                ctx_type[b, position] = CONTEXT_FIELDS.index(field) + 1
                ctx_value[b, position] = self.ctx_vocab[field].get(value, 0)
                token_mask[b, position] = True
                token_node[b, position] = 0
                position += 1

            numeric[b] = (
                material.ph / 7.0,
                np.log10(material.surface_area + 1.0) / 2.5,
                material.strain * 50.0,
                material.defect,
                _heterojunction_flag(material.heterojunction),
                material.dopant_fraction * 10.0,
                np.log10(max(material.cocatalyst_loading, 0.0) * 1000.0 + 1.0) / 2.0,
                np.log10(max(material.light_intensity, 0.0) + 1.0) / 2.5,
                np.log10(max(material.catalyst_loading, 0.0) * 1000.0 + 1.0) / 2.0,
                np.log10(max(material.sacrificial_concentration, 0.0) * 100.0 + 1.0) / 2.0,
            )
            targets[b] = (material.eg, np.log1p(material.her), material.stability)
            ids.append(material.id)
            labels.append(material.label)

        # Row-normalise adjacency; give padding rows a self-loop.
        for b in range(batch_size):
            active = node_mask[b]
            adjacency[b, ~active, :] = 0.0
            adjacency[b, :, ~active] = 0.0
            adjacency[b][np.diag_indices(num_nodes)] += 1.0
            degree = adjacency[b].sum(axis=1, keepdims=True)
            adjacency[b] = adjacency[b] / np.maximum(degree, 1e-9)

        node_select = np.eye(num_nodes)[token_node]  # (B, T, N)
        node_select *= token_mask[..., None]
        pair_bias = node_select @ adjacency @ np.swapaxes(node_select, -1, -2)

        return FeatureBatch(
            token_type=token_type,
            element_id=element_id,
            role_id=role_id,
            stoich_feat=stoich_feat,
            ctx_type=ctx_type,
            ctx_value=ctx_value,
            token_mask=token_mask,
            token_node=token_node,
            node_select=node_select,
            node_features=node_features,
            node_mask=node_mask,
            adjacency=adjacency,
            pair_bias=pair_bias,
            numeric=numeric,
            targets=targets,
            ids=ids,
            labels=labels,
        )

    # --------------------------------------------------------------------- I/O
    def to_dict(self) -> dict[str, object]:
        return {
            "ctx_vocab": self.ctx_vocab,
            "material_mean": self.material_mean.tolist(),
            "material_std": self.material_std.tolist(),
            "max_elements": self.max_elements,
            "context_fields": list(CONTEXT_FIELDS),
            "semantic_version": 1,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Featurizer":
        if payload.get("semantic_version") != 1:
            raise ValueError("unsupported featurizer version")
        if tuple(payload.get("context_fields", ())) != CONTEXT_FIELDS:
            raise ValueError("featurizer context fields do not match this version")
        return cls(
            ctx_vocab={k: dict(v) for k, v in payload["ctx_vocab"].items()},  # type: ignore[union-attr]
            material_mean=np.asarray(payload["material_mean"], dtype=np.float64),
            material_std=np.asarray(payload["material_std"], dtype=np.float64),
            max_elements=int(payload["max_elements"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Featurizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def vocab_size(self, field: str) -> int:
        """Number of ids including the unknown slot."""
        return max(self.ctx_vocab[field].values(), default=0) + 1


_RAW_ELEMENT_TABLE = element_feature_matrix()


def _raw_element_row(element_id: int) -> np.ndarray:
    return _RAW_ELEMENT_TABLE[element_id]


def _material_descriptor(material: Material) -> np.ndarray:
    return np.asarray([material.descriptors[name] for name in MATERIAL_FEATURES], dtype=np.float64)
