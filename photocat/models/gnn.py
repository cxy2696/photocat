"""Message-passing encoder over the material's knowledge-graph subgraph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..autograd import Tensor, as_tensor, linear, save_params, load_params
from ..data.dataset import NODE_FEATURES


@dataclass
class GNNOutput:
    node_embeddings: Tensor  # (B, N, H)
    material: Tensor  # (B, H)
    pooled: Tensor  # (B, H)


def _xavier(rng: np.random.Generator, fan_in: int, fan_out: int) -> Tensor:
    scale = np.sqrt(2.0 / (fan_in + fan_out))
    return Tensor(rng.normal(0.0, scale, (fan_in, fan_out)), requires_grad=True)


class KGSubgraphGNN:
    """Two-layer GCN over ``material -> element`` and element co-occurrence edges.

    The GNN is trained jointly with the transformer.  Its node embeddings are
    injected into the corresponding element tokens, and its pooled material
    embedding initialises the ``[CLS]`` token.  This is the "KG-derived node
    embedding" path in the proposal's Figure 1.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        layers: int = 2,
        in_dim: int = NODE_FEATURES,
        seed: int = 0,
    ) -> None:
        if layers < 1:
            raise ValueError("layers must be >= 1")
        rng = np.random.default_rng(seed)
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.in_dim = in_dim
        self.weights: list[Tensor] = []
        self.biases: list[Tensor] = []
        for layer in range(layers):
            fan_in = in_dim if layer == 0 else hidden_dim
            self.weights.append(_xavier(rng, fan_in, hidden_dim))
            self.biases.append(Tensor(np.zeros(hidden_dim), requires_grad=True))

    # ------------------------------------------------------------------ params
    def named_params(self) -> dict[str, Tensor]:
        params: dict[str, Tensor] = {}
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            params[f"gnn.W{i}"] = weight
            params[f"gnn.b{i}"] = bias
        return params

    def num_params(self) -> int:
        return sum(param.data.size for param in self.named_params().values())

    def save(self, path: str) -> None:
        save_params(path, self.named_params())

    def load(self, path: str) -> None:
        load_params(path, self.named_params())

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        node_features: np.ndarray | Tensor,
        adjacency: np.ndarray | Tensor,
        node_mask: np.ndarray,
    ) -> GNNOutput:
        x = as_tensor(node_features) if not isinstance(node_features, Tensor) else node_features
        a = as_tensor(adjacency) if not isinstance(adjacency, Tensor) else adjacency
        mask = np.asarray(node_mask, dtype=np.float64)
        mask_t = Tensor(mask)
        mask_3d = mask_t.reshape(mask.shape[0], mask.shape[1], 1)
        hidden = x
        for weight, bias in zip(self.weights, self.biases):
            hidden = a @ hidden
            hidden = linear(hidden, weight, bias)
            hidden = hidden.relu()
            hidden = hidden * mask_3d

        material_mask = np.zeros((mask.shape[0], mask.shape[1], 1), dtype=np.float64)
        material_mask[:, 0, 0] = 1.0
        material = (hidden * Tensor(material_mask)).sum(axis=1)  # (B, H)
        counts = mask.sum(axis=1, keepdims=True)
        pooled = (hidden * mask_3d).sum(axis=1) / np.maximum(counts, 1.0)
        return GNNOutput(node_embeddings=hidden, material=material, pooled=pooled)
