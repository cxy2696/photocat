"""Graph-augmented transformer for bandgap, HER and stability prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..autograd import (
    Tensor,
    as_tensor,
    concat,
    embedding_lookup,
    layernorm,
    linear,
    load_params,
    save_params,
)
from ..data.dataset import CONTEXT_FIELDS, MAX_TOKENS, NODE_FEATURES, NUMERIC_FEATURES, FeatureBatch
from .gnn import GNNOutput, KGSubgraphGNN


@dataclass
class ModelOutput:
    eg: Tensor  # standardised bandgap
    her: Tensor  # standardised log1p(HER)
    stability: Tensor  # stability logit
    attention: list[np.ndarray] | None = None

    def predict(self) -> np.ndarray:
        """Stack the three heads into an (B, 3) array (model units)."""
        return np.stack([self.eg.data, self.her.data, self.stability.data], axis=-1)


def _xavier(rng: np.random.Generator, fan_in: int, fan_out: int) -> Tensor:
    scale = np.sqrt(2.0 / (fan_in + fan_out))
    return Tensor(rng.normal(0.0, scale, (fan_in, fan_out)), requires_grad=True)


def _linear_pair(rng: np.random.Generator, fan_in: int, fan_out: int) -> tuple[Tensor, Tensor]:
    return _xavier(rng, fan_in, fan_out), Tensor(np.zeros(fan_out), requires_grad=True)


class GraphAugmentedTransformer:
    """Transformer whose tokens are augmented with KG/GNN node embeddings.

    Design
    ------
    * ``[CLS]`` + up to six element tokens + seven context tokens.
    * Each element token = element embedding + role embedding + projected GNN
      node embedding + composition features + position.
    * Attention logits receive a per-head bias proportional to the subgraph
      adjacency between the two tokens' nodes (the "graph-augmented" part).
    * Two regression heads share a trunk: bandgap (eV), log1p(HER), stability
      logit.  Targets are standardised by the trainer.
    """

    def __init__(
        self,
        element_vocab_size: int,
        ctx_sizes: Sequence[int],
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        gnn_dim: int = 32,
        gnn_layers: int = 2,
        max_tokens: int = MAX_TOKENS,
        numeric_dim: int = len(NUMERIC_FEATURES),
        node_feature_dim: int = NODE_FEATURES,
        seed: int = 0,
    ) -> None:
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if len(ctx_sizes) != len(CONTEXT_FIELDS):
            raise ValueError(f"expected {len(CONTEXT_FIELDS)} context vocab sizes")
        rng = np.random.default_rng(seed)
        self.element_vocab_size = element_vocab_size
        self.ctx_sizes = tuple(int(size) for size in ctx_sizes)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.gnn_dim = gnn_dim
        self.gnn_layers = gnn_layers
        self.max_tokens = max_tokens
        self.numeric_dim = numeric_dim
        self.node_feature_dim = node_feature_dim

        self.gnn = KGSubgraphGNN(
            hidden_dim=gnn_dim, layers=gnn_layers, in_dim=node_feature_dim, seed=seed + 101
        )

        # Token embeddings.
        self.element_embedding = Tensor(
            rng.normal(0.0, 0.08, (element_vocab_size, d_model)), requires_grad=True
        )
        self.role_embedding = Tensor(rng.normal(0.0, 0.05, (5, d_model)), requires_grad=True)
        self.context_type_embedding = Tensor(
            rng.normal(0.0, 0.05, (len(CONTEXT_FIELDS) + 1, d_model)), requires_grad=True
        )
        offsets = np.zeros(len(CONTEXT_FIELDS) + 1, dtype=np.int64)
        running = 0
        for i, size in enumerate(self.ctx_sizes, start=1):
            offsets[i] = running
            running += size
        self.ctx_offsets = offsets
        self.context_value_embedding = Tensor(
            rng.normal(0.0, 0.05, (max(running, 1), d_model)), requires_grad=True
        )
        self.position_embedding = Tensor(
            rng.normal(0.0, 0.02, (max_tokens, d_model)), requires_grad=True
        )
        self.cls_embedding = Tensor(rng.normal(0.0, 0.02, (d_model,)), requires_grad=True)

        self.W_stoich, self.b_stoich = _linear_pair(rng, 2, d_model)
        self.W_node, self.b_node = _linear_pair(rng, gnn_dim, d_model)
        self.W_material, self.b_material = _linear_pair(rng, gnn_dim, d_model)
        self.W_numeric, self.b_numeric = _linear_pair(rng, numeric_dim, d_model)

        # Transformer blocks.
        self.blocks: list[dict[str, Tensor]] = []
        for _ in range(n_layers):
            block: dict[str, Tensor] = {
                "ln1_g": Tensor(np.ones(d_model), requires_grad=True),
                "ln1_b": Tensor(np.zeros(d_model), requires_grad=True),
                "ln2_g": Tensor(np.ones(d_model), requires_grad=True),
                "ln2_b": Tensor(np.zeros(d_model), requires_grad=True),
                "edge_w": Tensor(np.zeros(n_heads), requires_grad=True),
            }
            block["Wq"], block["bq"] = _linear_pair(rng, d_model, d_model)
            block["Wk"], block["bk"] = _linear_pair(rng, d_model, d_model)
            block["Wv"], block["bv"] = _linear_pair(rng, d_model, d_model)
            block["Wo"], block["bo"] = _linear_pair(rng, d_model, d_model)
            block["W1"], block["b1"] = _linear_pair(rng, d_model, d_ff)
            block["W2"], block["b2"] = _linear_pair(rng, d_ff, d_model)
            self.blocks.append(block)

        # Shared head.
        self.ln_f_g = Tensor(np.ones(d_model), requires_grad=True)
        self.ln_f_b = Tensor(np.zeros(d_model), requires_grad=True)
        self.W_head1, self.b_head1 = _linear_pair(rng, 2 * d_model, d_model)
        head_scale = np.sqrt(1.0 / d_model)
        self.W_head_eg = Tensor(rng.normal(0.0, head_scale, (d_model, 1)), requires_grad=True)
        self.b_head_eg = Tensor(np.zeros(1), requires_grad=True)
        self.W_head_her = Tensor(rng.normal(0.0, head_scale, (d_model, 1)), requires_grad=True)
        self.b_head_her = Tensor(np.zeros(1), requires_grad=True)
        self.W_head_stab = Tensor(rng.normal(0.0, head_scale, (d_model, 1)), requires_grad=True)
        self.b_head_stab = Tensor(np.zeros(1), requires_grad=True)

    # ------------------------------------------------------------------ params
    def named_params(self) -> dict[str, Tensor]:
        params: dict[str, Tensor] = {}
        params.update(self.gnn.named_params())
        params.update(
            {
                "tok.element": self.element_embedding,
                "tok.role": self.role_embedding,
                "tok.ctx_type": self.context_type_embedding,
                "tok.ctx_value": self.context_value_embedding,
                "tok.position": self.position_embedding,
                "tok.cls": self.cls_embedding,
                "proj.stoich.W": self.W_stoich,
                "proj.stoich.b": self.b_stoich,
                "proj.node.W": self.W_node,
                "proj.node.b": self.b_node,
                "proj.material.W": self.W_material,
                "proj.material.b": self.b_material,
                "proj.numeric.W": self.W_numeric,
                "proj.numeric.b": self.b_numeric,
                "head.ln_g": self.ln_f_g,
                "head.ln_b": self.ln_f_b,
                "head.W1": self.W_head1,
                "head.b1": self.b_head1,
                "head.eg.W": self.W_head_eg,
                "head.eg.b": self.b_head_eg,
                "head.her.W": self.W_head_her,
                "head.her.b": self.b_head_her,
                "head.stab.W": self.W_head_stab,
                "head.stab.b": self.b_head_stab,
            }
        )
        for i, block in enumerate(self.blocks):
            for name, param in block.items():
                params[f"block{i}.{name}"] = param
        return params

    def num_params(self) -> int:
        return sum(param.data.size for param in self.named_params().values())

    def config(self) -> dict[str, object]:
        return {
            "element_vocab_size": self.element_vocab_size,
            "ctx_sizes": list(self.ctx_sizes),
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "gnn_dim": self.gnn_dim,
            "gnn_layers": self.gnn_layers,
            "max_tokens": self.max_tokens,
            "numeric_dim": self.numeric_dim,
            "node_feature_dim": self.node_feature_dim,
        }

    @classmethod
    def from_config(cls, config: dict[str, object], seed: int = 0) -> "GraphAugmentedTransformer":
        return cls(
            element_vocab_size=int(config["element_vocab_size"]),
            ctx_sizes=[int(x) for x in config["ctx_sizes"]],  # type: ignore[union-attr]
            d_model=int(config["d_model"]),
            n_heads=int(config["n_heads"]),
            n_layers=int(config["n_layers"]),
            d_ff=int(config["d_ff"]),
            gnn_dim=int(config["gnn_dim"]),
            gnn_layers=int(config["gnn_layers"]),
            max_tokens=int(config["max_tokens"]),
            numeric_dim=int(config["numeric_dim"]),
            node_feature_dim=int(config["node_feature_dim"]),
            seed=seed,
        )

    def save(self, path: str) -> None:
        save_params(path, self.named_params())

    def load(self, path: str) -> None:
        load_params(path, self.named_params())

    # ----------------------------------------------------------------- forward
    def forward(self, batch: FeatureBatch, return_attention: bool = False) -> ModelOutput:
        token_type = batch.token_type
        is_cls = Tensor((token_type == 1)[..., None].astype(np.float64))
        is_element = Tensor(
            ((token_type >= 2) & (token_type <= 5))[..., None].astype(np.float64)
        )
        is_context = Tensor((token_type == 6)[..., None].astype(np.float64))

        gnn_out: GNNOutput = self.gnn.forward(
            batch.node_features, batch.adjacency, batch.node_mask
        )

        element_emb = embedding_lookup(self.element_embedding, batch.element_id) * is_element
        role_emb = embedding_lookup(self.role_embedding, batch.role_id) * is_element
        ctx_index = batch.ctx_value + self.ctx_offsets[batch.ctx_type]
        context_emb = embedding_lookup(self.context_value_embedding, ctx_index) * is_context
        context_type_emb = (
            embedding_lookup(self.context_type_embedding, batch.ctx_type) * is_context
        )

        stoich = linear(as_tensor(batch.stoich_feat), self.W_stoich, self.b_stoich)
        node_tokens = self._select_nodes(batch.node_select, gnn_out.node_embeddings)
        node_tokens = linear(node_tokens, self.W_node, self.b_node) * is_element

        hidden = (
            element_emb
            + role_emb
            + context_emb
            + context_type_emb
            + stoich
            + node_tokens
            + self.position_embedding
        )
        hidden = hidden + self.cls_embedding * is_cls
        numeric = linear(as_tensor(batch.numeric), self.W_numeric, self.b_numeric)
        material = linear(gnn_out.material, self.W_material, self.b_material)
        cls_input = (numeric + material).reshape(batch.token_type.shape[0], 1, self.d_model)
        hidden = hidden + cls_input * is_cls

        pad_mask = ~batch.token_mask
        key_pad = pad_mask[:, None, None, :]
        pair_bias = Tensor(batch.pair_bias).reshape(
            batch.token_type.shape[0], 1, batch.token_type.shape[1], batch.token_type.shape[1]
        )
        attention_maps: list[np.ndarray] = []

        for block in self.blocks:
            x = layernorm(hidden, block["ln1_g"], block["ln1_b"])
            q = self._heads(linear(x, block["Wq"], block["bq"]))
            k = self._heads(linear(x, block["Wk"], block["bk"]))
            v = self._heads(linear(x, block["Wv"], block["bv"]))
            scores = (q @ k.swapaxes(-1, -2)) * (1.0 / np.sqrt(self.d_model // self.n_heads))
            bias = pair_bias * block["edge_w"].reshape(1, self.n_heads, 1, 1)
            scores = scores + bias
            scores = scores.masked_fill(key_pad, -1e9)
            attention = scores.softmax(axis=-1)
            context = attention @ v
            context = context.transpose(0, 2, 1, 3).reshape(
                batch.token_type.shape[0], self.max_tokens, self.d_model
            )
            hidden = hidden + linear(context, block["Wo"], block["bo"])

            y = layernorm(hidden, block["ln2_g"], block["ln2_b"])
            y = linear(y, block["W1"], block["b1"]).gelu()
            y = linear(y, block["W2"], block["b2"])
            hidden = hidden + y
            if return_attention:
                attention_maps.append(np.asarray(attention.data))

        final = layernorm(hidden, self.ln_f_g, self.ln_f_b)
        cls_vector = (final * is_cls).sum(axis=1)
        counts = np.maximum(batch.token_mask.sum(axis=1, keepdims=True), 1)
        pooled = (final * Tensor(batch.token_mask[..., None].astype(np.float64))).sum(
            axis=1
        ) / counts
        features = concat([cls_vector, pooled], axis=-1)
        features = linear(features, self.W_head1, self.b_head1).gelu()
        eg = linear(features, self.W_head_eg, self.b_head_eg).reshape(-1)
        her = linear(features, self.W_head_her, self.b_head_her).reshape(-1)
        stability = linear(features, self.W_head_stab, self.b_head_stab).reshape(-1)
        return ModelOutput(
            eg=eg,
            her=her,
            stability=stability,
            attention=attention_maps if return_attention else None,
        )

    # ------------------------------------------------------------------ helpers
    def _heads(self, tensor: Tensor) -> Tensor:
        batch = tensor.shape[0]
        head_dim = self.d_model // self.n_heads
        return tensor.reshape(batch, self.max_tokens, self.n_heads, head_dim).transpose(0, 2, 1, 3)

    @staticmethod
    def _select_nodes(node_select: np.ndarray, node_embeddings: Tensor) -> Tensor:
        return as_tensor(node_select) @ node_embeddings
