"""Optional PyTorch training/prediction path for the composition model."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .data.dataset import Featurizer
from .data.elements import ELEMENT_IDS
from .data.synth import Material
from .metrics import pearson, rmse
from .models.ensemble import EnsemblePrediction
from .models.torch_transformer import TORCH_AVAILABLE, TorchGraphAugmentedTransformer
from .training import (
    TargetScaler,
    TrainConfig,
    TrainingHistory,
    family_sample_weights,
)

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F
else:  # pragma: no cover
    torch = None
    F = None


@dataclass
class TorchTrainingResult:
    model: object
    scaler: TargetScaler
    history: TrainingHistory
    config: TrainConfig
    device: str = "cpu"


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is not installed. Install the optional extra with "
            '`pip install -e ".[torch]"`.'
        )


def batch_to_torch(batch, device: str = "cpu", dtype=None) -> dict[str, object]:
    _require_torch()
    if dtype is None:
        dtype = torch.float32
    long_fields = ("token_type", "element_id", "role_id", "ctx_type", "ctx_value")
    float_fields = (
        "stoich_feat",
        "token_node",
        "node_select",
        "node_features",
        "adjacency",
        "pair_bias",
        "numeric",
    )
    bool_fields = ("token_mask", "node_mask")
    result: dict[str, object] = {}
    for name in long_fields:
        result[name] = torch.as_tensor(getattr(batch, name), dtype=torch.long, device=device)
    for name in float_fields:
        result[name] = torch.as_tensor(getattr(batch, name), dtype=dtype, device=device)
    for name in bool_fields:
        result[name] = torch.as_tensor(getattr(batch, name), dtype=torch.bool, device=device)
    return result


def forward_torch_batches(
    model: TorchGraphAugmentedTransformer,
    batch,
    batch_size: int = 256,
    return_attention: bool = False,
    device: str = "cpu",
) -> dict[str, object]:
    _require_torch()
    model.eval()
    raw: list[np.ndarray] = []
    attention: list[list[np.ndarray]] | None = [] if return_attention else None
    with torch.no_grad():
        for start in range(0, len(batch), batch_size):
            chunk = batch.take(np.arange(start, min(start + batch_size, len(batch))))
            inputs = batch_to_torch(chunk, device=device, dtype=model.dtype)
            output = model(inputs, return_attention=return_attention)
            raw.append(
                np.stack(
                    [
                        output.eg.detach().cpu().numpy(),
                        output.her.detach().cpu().numpy(),
                        output.stability.detach().cpu().numpy(),
                    ],
                    axis=-1,
                )
            )
            if attention is not None and output.attention is not None:
                if not attention:
                    attention = [[] for _ in output.attention]
                for layer, values in enumerate(output.attention):
                    attention[layer].append(values.cpu().numpy())
    result: dict[str, object] = {"raw": np.concatenate(raw, axis=0)}
    if attention is not None:
        result["attention"] = [np.concatenate(layer, axis=0) for layer in attention]
    return result


def predict_torch_materials(
    results: Sequence[TorchTrainingResult],
    featurizer: Featurizer,
    materials: Sequence[Material],
    device: str = "cpu",
) -> EnsemblePrediction:
    batch = featurizer.transform(materials)
    members = []
    for result in results:
        raw = np.asarray(
            forward_torch_batches(result.model, batch, device=device)["raw"]
        )
        members.append(result.scaler.to_physical(raw))
    stacked = np.stack(members, axis=0)
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


class TorchTrainer:
    def __init__(self, config: TrainConfig | None = None, device: str = "cpu", dtype=None) -> None:
        _require_torch()
        self.config = config or TrainConfig()
        self.device = device
        self.dtype = dtype if dtype is not None else torch.float32

    def train(
        self,
        model: TorchGraphAugmentedTransformer,
        featurizer: Featurizer,
        train_materials: Sequence[Material],
        val_materials: Sequence[Material],
        seed: int | None = None,
    ) -> TorchTrainingResult:
        config = self.config
        model.to(self.device)
        rng = np.random.default_rng(config.seed if seed is None else seed)
        train_batch = featurizer.transform(train_materials)
        val_batch = featurizer.transform(val_materials)
        scaler = TargetScaler.fit(train_batch.targets)
        scaled = scaler.scale(train_batch.targets)
        masks = np.isfinite(train_batch.targets)
        safe_scaled = np.nan_to_num(scaled, nan=0.0)
        safe_raw = np.nan_to_num(train_batch.targets, nan=0.0)

        sample_weights = np.ones(len(train_materials), dtype=np.float64)
        sampling_probabilities = np.full(len(train_materials), 1.0 / len(train_materials))
        if config.mitigation == "balanced":
            sample_weights = family_sample_weights(train_materials)
        elif config.mitigation == "oversample":
            sample_weights = family_sample_weights(train_materials)
            sampling_probabilities = sample_weights / sample_weights.sum()
        elif config.mitigation != "none":
            raise ValueError(f"unknown mitigation strategy {config.mitigation!r}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        history = TrainingHistory()
        best_state = copy.deepcopy(model.state_dict())
        best_val = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        started = time.time()

        for epoch in range(1, config.epochs + 1):
            for group in optimizer.param_groups:
                group["lr"] = config.lr * (
                    0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / config.epochs))
                )
            if config.mitigation == "oversample":
                order = rng.choice(
                    len(train_materials),
                    size=len(train_materials),
                    replace=True,
                    p=sampling_probabilities,
                )
            else:
                order = rng.permutation(len(train_materials))
            model.train()
            total_loss = 0.0
            seen = 0
            for start in range(0, len(order), config.batch_size):
                index = order[start : start + config.batch_size]
                chunk = train_batch.take(index)
                inputs = batch_to_torch(chunk, device=self.device, dtype=self.dtype)
                output = model(inputs)
                weights_np = sample_weights[index]
                weights = torch.as_tensor(weights_np, dtype=self.dtype, device=self.device)
                mask_t = torch.as_tensor(masks[index], dtype=torch.bool, device=self.device)
                target_t = torch.as_tensor(safe_scaled[index], dtype=self.dtype, device=self.device)
                raw_t = torch.as_tensor(safe_raw[index], dtype=self.dtype, device=self.device)
                losses = []
                for task, (head, weight) in enumerate(
                    zip((output.eg, output.her), config.loss_weights[:2])
                ):
                    task_weight = weights * mask_t[:, task].to(self.dtype)
                    if float(task_weight.sum()) > 0:
                        error = F.mse_loss(head, target_t[:, task], reduction="none")
                        losses.append(weight * (error * task_weight).sum() / task_weight.sum())
                stability_weight = weights * mask_t[:, 2].to(self.dtype)
                if float(stability_weight.sum()) > 0:
                    bce = F.binary_cross_entropy_with_logits(
                        output.stability, raw_t[:, 2], reduction="none"
                    )
                    losses.append(
                        config.loss_weights[2]
                        * (bce * stability_weight).sum()
                        / stability_weight.sum()
                    )
                if not losses:
                    continue
                loss = losses[0]
                for term in losses[1:]:
                    loss = loss + term
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                total_loss += float(loss.detach()) * len(index)
                seen += len(index)

            train_loss = total_loss / max(seen, 1)
            val_raw = np.asarray(
                forward_torch_batches(model, val_batch, batch_size=512, device=self.device)["raw"]
            )
            scaled_val = scaler.scale(val_batch.targets)
            val_masks = np.isfinite(val_batch.targets)
            val_losses: list[float] = []
            for task in range(2):
                if val_masks[:, task].any():
                    val_losses.append(
                        float(
                            np.mean(
                                (val_raw[val_masks[:, task], task] - scaled_val[val_masks[:, task], task]) ** 2
                            )
                        )
                    )
            if val_masks[:, 2].any():
                z = val_raw[val_masks[:, 2], 2]
                y = val_batch.targets[val_masks[:, 2], 2]
                val_losses.append(
                    0.25 * float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))))
                )
            val_loss = float(sum(val_losses)) if val_losses else float("inf")
            physical = scaler.to_physical(val_raw)
            val_eg = (
                rmse(val_batch.targets[val_masks[:, 0], 0], physical[val_masks[:, 0], 0])
                if val_masks[:, 0].any()
                else float("nan")
            )
            val_her = (
                pearson(val_batch.targets[val_masks[:, 1], 1], physical[val_masks[:, 1], 1])
                if val_masks[:, 1].any()
                else float("nan")
            )
            history.epochs.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_eg_rmse": float(val_eg),
                    "val_her_pearson": float(val_her),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1
            if config.verbose:
                parts = [f"epoch {epoch:3d}", f"train {train_loss:.4f}", f"val {val_loss:.4f}"]
                if np.isfinite(val_eg):
                    parts.append(f"Eg RMSE {val_eg:.3f} eV")
                if np.isfinite(val_her):
                    parts.append(f"HER r {val_her:.3f}")
                print(" | ".join(parts), flush=True)
            if epochs_without_improvement >= config.patience:
                break

        model.load_state_dict(best_state)
        history.best_epoch = best_epoch
        history.best_val_loss = best_val
        history.seconds = time.time() - started
        return TorchTrainingResult(
            model=model,
            scaler=scaler,
            history=history,
            config=config,
            device=self.device,
        )


def train_torch_ensemble(
    configs: Sequence[TrainConfig],
    featurizer: Featurizer,
    train_materials: Sequence[Material],
    val_materials: Sequence[Material],
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 128,
    gnn_dim: int = 32,
    dropout: float = 0.1,
    device: str = "cpu",
    dtype=None,
    verbose: bool = True,
) -> list[TorchTrainingResult]:
    _require_torch()
    if not configs:
        raise ValueError("at least one training config is required")
    ctx_sizes = [featurizer.vocab_size(field) for field in featurizer.ctx_vocab]
    element_vocab = max(ELEMENT_IDS.values()) + 1
    results: list[TorchTrainingResult] = []
    for index, config in enumerate(configs):
        model = TorchGraphAugmentedTransformer(
            element_vocab_size=element_vocab,
            ctx_sizes=ctx_sizes,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            gnn_dim=gnn_dim,
            dropout=dropout,
            dtype=dtype,
            seed=config.seed,
        )
        if verbose:
            print(
                f"\n[torch ensemble {index + 1}/{len(configs)}] seed={config.seed}",
                flush=True,
            )
        trainer = TorchTrainer(config, device=device, dtype=dtype)
        results.append(trainer.train(model, featurizer, train_materials, val_materials))
    return results
