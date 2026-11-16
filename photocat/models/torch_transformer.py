"""PyTorch implementation of the composition model.

This mirrors :mod:`photocat.models.transformer` operation for operation.  The
optional ``load_from_numpy`` method copies the NumPy weights into the Torch
module; tests use it to check that both implementations give the same output
when dropout is disabled.  PyTorch is an optional dependency; importing this
module without it raises a clear error only when a model is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.dataset import CONTEXT_FIELDS, MAX_TOKENS, NODE_FEATURES, NUMERIC_FEATURES

try:  # pragma: no cover - exercised only when torch is installed
    import torch
    import torch.nn.functional as F
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


_BaseModule = nn.Module if TORCH_AVAILABLE else object


@dataclass
class TorchOutput:
    eg: Any
    her: Any
    stability: Any
    attention: list[Any] | None = None


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. Install the optional extra with "
            '`pip install -e ".[torch]"`, or use backend="numpy".'
        )


class TorchKGSubgraphGNN(_BaseModule):
    def __init__(self, hidden_dim: int, layers: int, in_dim: int, dtype: Any) -> None:
        _require_torch()
        super().__init__()
        self.layers = nn.ModuleList()
        for layer in range(layers):
            self.layers.append(nn.Linear(in_dim if layer == 0 else hidden_dim, hidden_dim, dtype=dtype))

    def forward(self, node_features, adjacency, node_mask):
        mask = node_mask.to(node_features.dtype).unsqueeze(-1)
        hidden = node_features
        for layer in self.layers:
            hidden = torch.bmm(adjacency, hidden)
            hidden = torch.relu(layer(hidden)) * mask
        material = hidden[:, 0]
        return hidden, material


class TorchBlock(_BaseModule):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, dtype: Any) -> None:
        _require_torch()
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model, dtype=dtype)
        self.ln2 = nn.LayerNorm(d_model, dtype=dtype)
        self.q = nn.Linear(d_model, d_model, dtype=dtype)
        self.k = nn.Linear(d_model, d_model, dtype=dtype)
        self.v = nn.Linear(d_model, d_model, dtype=dtype)
        self.o = nn.Linear(d_model, d_model, dtype=dtype)
        self.edge_w = nn.Parameter(torch.zeros(n_heads, dtype=dtype))
        self.ff1 = nn.Linear(d_model, d_ff, dtype=dtype)
        self.ff2 = nn.Linear(d_ff, d_model, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, pair_bias, key_pad):
        batch, tokens, _ = hidden.shape
        x = self.ln1(hidden)
        q = self.q(x).view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.head_dim)
        bias = pair_bias.unsqueeze(1) * self.edge_w.view(1, self.n_heads, 1, 1)
        scores = scores + bias
        scores = scores.masked_fill(key_pad, -1e9)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(batch, tokens, -1)
        hidden = hidden + self.o(context)
        y = self.ln2(hidden)
        y = self.ff2(F.gelu(self.ff1(y), approximate="tanh"))
        hidden = hidden + self.dropout(y)
        return hidden, attention


class TorchGraphAugmentedTransformer(_BaseModule):
    """Drop-in PyTorch counterpart of ``GraphAugmentedTransformer``."""

    def __init__(
        self,
        element_vocab_size: int,
        ctx_sizes: list[int] | tuple[int, ...],
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        gnn_dim: int = 32,
        gnn_layers: int = 2,
        max_tokens: int = MAX_TOKENS,
        numeric_dim: int = len(NUMERIC_FEATURES),
        node_feature_dim: int = NODE_FEATURES,
        dropout: float = 0.1,
        dtype: Any = None,
        seed: int = 0,
    ) -> None:
        _require_torch()
        super().__init__()
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype
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
        self.dropout_p = dropout
        torch.manual_seed(seed)

        self.gnn = TorchKGSubgraphGNN(gnn_dim, gnn_layers, node_feature_dim, dtype)
        self.element_embedding = nn.Embedding(element_vocab_size, d_model, dtype=dtype)
        self.role_embedding = nn.Embedding(5, d_model, dtype=dtype)
        self.context_type_embedding = nn.Embedding(len(CONTEXT_FIELDS) + 1, d_model, dtype=dtype)
        offsets = np.zeros(len(CONTEXT_FIELDS) + 1, dtype=np.int64)
        running = 0
        for index, size in enumerate(self.ctx_sizes, start=1):
            offsets[index] = running
            running += size
        self.register_buffer("ctx_offsets", torch.as_tensor(offsets, dtype=torch.long))
        self.context_value_embedding = nn.Embedding(max(running, 1), d_model, dtype=dtype)
        self.position_embedding = nn.Parameter(torch.zeros(max_tokens, d_model, dtype=dtype))
        self.cls_embedding = nn.Parameter(torch.zeros(d_model, dtype=dtype))
        self.stoich_proj = nn.Linear(2, d_model, dtype=dtype)
        self.node_proj = nn.Linear(gnn_dim, d_model, dtype=dtype)
        self.material_proj = nn.Linear(gnn_dim, d_model, dtype=dtype)
        self.numeric_proj = nn.Linear(numeric_dim, d_model, dtype=dtype)
        self.blocks = nn.ModuleList(
            [TorchBlock(d_model, n_heads, d_ff, dropout, dtype) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model, dtype=dtype)
        self.head1 = nn.Linear(2 * d_model, d_model, dtype=dtype)
        self.eg_head = nn.Linear(d_model, 1, dtype=dtype)
        self.her_head = nn.Linear(d_model, 1, dtype=dtype)
        self.stability_head = nn.Linear(d_model, 1, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for embedding, scale in (
            (self.element_embedding, 0.08),
            (self.role_embedding, 0.05),
            (self.context_type_embedding, 0.05),
            (self.context_value_embedding, 0.05),
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=scale)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_embedding, mean=0.0, std=0.02)
        for module in (
            self.stoich_proj,
            self.node_proj,
            self.material_proj,
            self.numeric_proj,
            self.head1,
            self.eg_head,
            self.her_head,
            self.stability_head,
        ):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
        for block in self.blocks:
            for module in (block.q, block.k, block.v, block.o, block.ff1, block.ff2):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
            nn.init.zeros_(block.edge_w)
        # Match the NumPy initialisation of the GNN layers.
        for layer in self.gnn.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

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
            "dropout": self.dropout_p,
        }

    @classmethod
    def from_config(cls, config: dict[str, object], seed: int = 0, dtype: Any = None):
        return cls(
            element_vocab_size=int(config["element_vocab_size"]),
            ctx_sizes=[int(size) for size in config["ctx_sizes"]],  # type: ignore[union-attr]
            d_model=int(config["d_model"]),
            n_heads=int(config["n_heads"]),
            n_layers=int(config["n_layers"]),
            d_ff=int(config["d_ff"]),
            gnn_dim=int(config["gnn_dim"]),
            gnn_layers=int(config["gnn_layers"]),
            max_tokens=int(config["max_tokens"]),
            numeric_dim=int(config["numeric_dim"]),
            node_feature_dim=int(config["node_feature_dim"]),
            dropout=float(config.get("dropout", 0.0)),
            dtype=dtype,
            seed=seed,
        )

    def forward(self, inputs: dict[str, Any], return_attention: bool = False) -> TorchOutput:
        token_type = inputs["token_type"]
        is_cls = (token_type == 1).to(self.dtype).unsqueeze(-1)
        is_element = ((token_type >= 2) & (token_type <= 5)).to(self.dtype).unsqueeze(-1)
        is_context = (token_type == 6).to(self.dtype).unsqueeze(-1)

        node_embeddings, material = self.gnn(
            inputs["node_features"], inputs["adjacency"], inputs["node_mask"]
        )
        element_emb = self.element_embedding(inputs["element_id"]) * is_element
        role_emb = self.role_embedding(inputs["role_id"]) * is_element
        ctx_index = inputs["ctx_value"] + self.ctx_offsets[inputs["ctx_type"]]
        context_emb = self.context_value_embedding(ctx_index) * is_context
        context_type_emb = self.context_type_embedding(inputs["ctx_type"]) * is_context
        stoich = self.stoich_proj(inputs["stoich_feat"])
        node_tokens = torch.bmm(inputs["node_select"], node_embeddings)
        node_tokens = self.node_proj(node_tokens) * is_element
        hidden = (
            element_emb
            + role_emb
            + context_emb
            + context_type_emb
            + stoich
            + node_tokens
            + self.position_embedding.unsqueeze(0)
        )
        hidden = hidden + self.cls_embedding * is_cls
        hidden = self.dropout(hidden)
        numeric = self.numeric_proj(inputs["numeric"])
        material = self.material_proj(material)
        cls_input = (numeric + material).unsqueeze(1)
        hidden = hidden + cls_input * is_cls

        key_pad = ~inputs["token_mask"][:, None, None, :]
        attention_maps: list[Any] = []
        for block in self.blocks:
            hidden, attention = block(hidden, inputs["pair_bias"], key_pad)
            if return_attention:
                attention_maps.append(attention.detach())
        final = self.ln_f(hidden)
        cls_vector = (final * is_cls).sum(dim=1)
        counts = inputs["token_mask"].sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (final * inputs["token_mask"].to(self.dtype).unsqueeze(-1)).sum(dim=1) / counts
        features = torch.cat([cls_vector, pooled], dim=-1)
        features = F.gelu(self.head1(features), approximate="tanh")
        eg = self.eg_head(features).squeeze(-1)
        her = self.her_head(features).squeeze(-1)
        stability = self.stability_head(features).squeeze(-1)
        return TorchOutput(
            eg=eg,
            her=her,
            stability=stability,
            attention=attention_maps if return_attention else None,
        )

    # ------------------------------------------------------------------ transfer
    def load_from_numpy(self, numpy_model: Any) -> "TorchGraphAugmentedTransformer":
        """Copy weights from a NumPy ``GraphAugmentedTransformer``.

        This is used by the equivalence test and by users who want to switch
        backends without retraining.
        """
        params = numpy_model.named_params()
        to_t = lambda name: torch.as_tensor(params[name].data, dtype=self.dtype)  # noqa: E731

        def copy_linear(module, prefix: str) -> None:
            module.weight.data.copy_(to_t(f"{prefix}.W").t())
            module.bias.data.copy_(to_t(f"{prefix}.b"))

        for index, layer in enumerate(self.gnn.layers):
            layer.weight.data.copy_(to_t(f"gnn.W{index}").t())
            layer.bias.data.copy_(to_t(f"gnn.b{index}"))
        self.element_embedding.weight.data.copy_(to_t("tok.element"))
        self.role_embedding.weight.data.copy_(to_t("tok.role"))
        self.context_type_embedding.weight.data.copy_(to_t("tok.ctx_type"))
        self.context_value_embedding.weight.data.copy_(to_t("tok.ctx_value"))
        self.position_embedding.data.copy_(to_t("tok.position"))
        self.cls_embedding.data.copy_(to_t("tok.cls"))
        copy_linear(self.stoich_proj, "proj.stoich")
        copy_linear(self.node_proj, "proj.node")
        copy_linear(self.material_proj, "proj.material")
        copy_linear(self.numeric_proj, "proj.numeric")
        for index, block in enumerate(self.blocks):
            prefix = f"block{index}"
            block.ln1.weight.data.copy_(to_t(f"{prefix}.ln1_g"))
            block.ln1.bias.data.copy_(to_t(f"{prefix}.ln1_b"))
            block.ln2.weight.data.copy_(to_t(f"{prefix}.ln2_g"))
            block.ln2.bias.data.copy_(to_t(f"{prefix}.ln2_b"))
            for name, module in (("q", block.q), ("k", block.k), ("v", block.v), ("o", block.o)):
                module.weight.data.copy_(to_t(f"{prefix}.W{name}").t())
                module.bias.data.copy_(to_t(f"{prefix}.b{name}"))
            block.edge_w.data.copy_(to_t(f"{prefix}.edge_w"))
            block.ff1.weight.data.copy_(to_t(f"{prefix}.W1").t())
            block.ff1.bias.data.copy_(to_t(f"{prefix}.b1"))
            block.ff2.weight.data.copy_(to_t(f"{prefix}.W2").t())
            block.ff2.bias.data.copy_(to_t(f"{prefix}.b2"))
        self.ln_f.weight.data.copy_(to_t("head.ln_g"))
        self.ln_f.bias.data.copy_(to_t("head.ln_b"))
        self.head1.weight.data.copy_(to_t("head.W1").t())
        self.head1.bias.data.copy_(to_t("head.b1"))
        self.eg_head.weight.data.copy_(to_t("head.eg.W").t())
        self.eg_head.bias.data.copy_(to_t("head.eg.b"))
        self.her_head.weight.data.copy_(to_t("head.her.W").t())
        self.her_head.bias.data.copy_(to_t("head.her.b"))
        self.stability_head.weight.data.copy_(to_t("head.stab.W").t())
        self.stability_head.bias.data.copy_(to_t("head.stab.b"))
        return self

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "config": self.config()}, path)

    @classmethod
    def load(cls, path: str, dtype: Any = None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = cls.from_config(payload["config"], dtype=dtype)
        model.load_state_dict(payload["state_dict"])
        return model
