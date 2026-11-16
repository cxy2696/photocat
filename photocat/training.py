"""Training loop, target scaling, batching and ensemble inference."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .autograd import Adam, bce_with_logits, mse_loss
from .data.dataset import FeatureBatch, Featurizer
from .data.synth import Material
from .metrics import pearson, rmse
from .models.transformer import GraphAugmentedTransformer


@dataclass
class TargetScaler:
    """Standardisation for bandgap and log-HER; stability stays a probability."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, targets: np.ndarray) -> "TargetScaler":
        targets = np.asarray(targets, dtype=np.float64)
        mean = targets.mean(axis=0)
        std = targets.std(axis=0)
        std[std < 1e-8] = 1.0
        mean[2] = 0.0
        std[2] = 1.0
        return cls(mean=mean, std=std)

    def scale(self, targets: np.ndarray) -> np.ndarray:
        return (np.asarray(targets, dtype=np.float64) - self.mean) / self.std

    def to_physical(self, predictions: np.ndarray) -> np.ndarray:
        predictions = np.asarray(predictions, dtype=np.float64)
        eg = predictions[:, 0] * self.std[0] + self.mean[0]
        her = np.expm1(predictions[:, 1] * self.std[1] + self.mean[1])
        stability = 1.0 / (1.0 + np.exp(-predictions[:, 2]))
        return np.stack([eg, her, stability], axis=-1)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> "TargetScaler":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
        )


@dataclass
class TrainConfig:
    epochs: int = 36
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 8
    grad_clip: float = 5.0
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 0.25)
    mitigation: str = "none"  # none | balanced | oversample
    verbose: bool = True
    seed: int = 0


@dataclass
class TrainingHistory:
    epochs: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    seconds: float = 0.0


@dataclass
class TrainingResult:
    model: GraphAugmentedTransformer
    scaler: TargetScaler
    history: TrainingHistory
    config: TrainConfig


def family_sample_weights(materials: Sequence[Material], power: float = 1.0) -> np.ndarray:
    """Inverse-frequency weights that balance crystal families."""
    families = np.asarray([material.family for material in materials])
    counts = {family: int((families == family).sum()) for family in set(families)}
    weights = np.asarray([1.0 / counts[family] ** power for family in families])
    weights /= weights.mean()
    return np.clip(weights, 0.2, 5.0)


def forward_batches(
    model: GraphAugmentedTransformer,
    batch: FeatureBatch,
    batch_size: int = 256,
    return_attention: bool = False,
) -> dict[str, object]:
    """Run the model over a batch in chunks; returns raw (model-unit) outputs."""
    raw_chunks: list[np.ndarray] = []
    attention_chunks: list[list[np.ndarray]] | None = [] if return_attention else None
    for start in range(0, len(batch), batch_size):
        chunk = batch.take(np.arange(start, min(start + batch_size, len(batch))))
        output = model.forward(chunk, return_attention=return_attention)
        raw_chunks.append(output.predict())
        if attention_chunks is not None and output.attention is not None:
            if not attention_chunks:
                attention_chunks = [[] for _ in output.attention]
            for layer, attention in enumerate(output.attention):
                attention_chunks[layer].append(attention)
    result: dict[str, object] = {"raw": np.concatenate(raw_chunks, axis=0)}
    if attention_chunks is not None:
        result["attention"] = [np.concatenate(layer, axis=0) for layer in attention_chunks]
    return result


def predict_materials(
    model: GraphAugmentedTransformer,
    featurizer: Featurizer,
    scaler: TargetScaler,
    materials: Sequence[Material],
    batch_size: int = 256,
    return_attention: bool = False,
) -> dict[str, object]:
    batch = featurizer.transform(materials)
    output = forward_batches(
        model, batch, batch_size=batch_size, return_attention=return_attention
    )
    physical = scaler.to_physical(output["raw"])  # type: ignore[arg-type]
    result: dict[str, object] = {
        "eg": physical[:, 0],
        "her": physical[:, 1],
        "stability": physical[:, 2],
        "raw": output["raw"],
        "batch": batch,
    }
    if "attention" in output:
        result["attention"] = output["attention"]
    return result


class Trainer:
    """Mini-batch Adam training with early stopping and family re-weighting."""

    def __init__(self, config: TrainConfig | None = None) -> None:
        self.config = config or TrainConfig()

    def train(
        self,
        model: GraphAugmentedTransformer,
        featurizer: Featurizer,
        train_materials: Sequence[Material],
        val_materials: Sequence[Material],
        seed: int | None = None,
    ) -> TrainingResult:
        config = self.config
        rng = np.random.default_rng(config.seed if seed is None else seed)
        train_batch = featurizer.transform(train_materials)
        val_batch = featurizer.transform(val_materials)
        scaler = TargetScaler.fit(train_batch.targets)
        scaled_targets = scaler.scale(train_batch.targets)

        sample_weights = np.ones(len(train_materials), dtype=np.float64)
        sampling_probabilities = np.full(len(train_materials), 1.0 / len(train_materials))
        if config.mitigation == "balanced":
            sample_weights = family_sample_weights(train_materials)
        elif config.mitigation == "oversample":
            sample_weights = family_sample_weights(train_materials)
            sampling_probabilities = sample_weights / sample_weights.sum()
        elif config.mitigation != "none":
            raise ValueError(f"unknown mitigation strategy {config.mitigation!r}")

        optimizer = Adam(
            model.named_params().values(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        history = TrainingHistory()
        best_state = {name: param.data.copy() for name, param in model.named_params().items()}
        best_val = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        started = time.time()

        for epoch in range(1, config.epochs + 1):
            # Cosine decay to 10% of the initial learning rate.
            optimizer.lr = config.lr * (
                0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / config.epochs))
            )
            if config.mitigation == "oversample":
                order = rng.choice(
                    len(train_materials), size=len(train_materials), replace=True,
                    p=sampling_probabilities,
                )
            else:
                order = rng.permutation(len(train_materials))

            train_loss_sum = 0.0
            seen = 0
            for start in range(0, len(order), config.batch_size):
                index = order[start : start + config.batch_size]
                chunk = train_batch.take(index)
                weights = sample_weights[index]
                output = model.forward(chunk)
                eg_loss = mse_loss(output.eg, scaled_targets[index, 0], weights)
                her_loss = mse_loss(output.her, scaled_targets[index, 1], weights)
                stability_loss = bce_with_logits(
                    output.stability, train_batch.targets[index, 2], weights
                )
                loss = (
                    config.loss_weights[0] * eg_loss
                    + config.loss_weights[1] * her_loss
                    + config.loss_weights[2] * stability_loss
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.clip_grad_norm(config.grad_clip)
                optimizer.step()
                train_loss_sum += float(loss.data) * len(index)
                seen += len(index)

            train_loss = train_loss_sum / max(seen, 1)
            val_raw = forward_batches(model, val_batch, batch_size=512)["raw"]
            val_raw = np.asarray(val_raw)
            val_loss, val_eg_rmse, val_her_pearson = self._validation_metrics(
                val_raw, val_batch, scaler
            )
            history.epochs.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_eg_rmse": float(val_eg_rmse),
                    "val_her_pearson": float(val_her_pearson),
                    "lr": float(optimizer.lr),
                }
            )
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    name: param.data.copy() for name, param in model.named_params().items()
                }
            else:
                epochs_without_improvement += 1
            if config.verbose:
                print(
                    f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | "
                    f"Eg RMSE {val_eg_rmse:.3f} eV | HER r {val_her_pearson:.3f}",
                    flush=True,
                )
            if epochs_without_improvement >= config.patience:
                break

        for name, param in model.named_params().items():
            param.data[...] = best_state[name]
        history.best_epoch = best_epoch
        history.best_val_loss = best_val
        history.seconds = time.time() - started
        return TrainingResult(model=model, scaler=scaler, history=history, config=config)

    @staticmethod
    def _validation_metrics(
        val_raw: np.ndarray,
        val_batch: FeatureBatch,
        scaler: TargetScaler,
    ) -> tuple[float, float, float]:
        scaled_val = scaler.scale(val_batch.targets)
        eg_mse = float(np.mean((val_raw[:, 0] - scaled_val[:, 0]) ** 2))
        her_mse = float(np.mean((val_raw[:, 1] - scaled_val[:, 1]) ** 2))
        z = val_raw[:, 2]
        y = val_batch.targets[:, 2]
        bce = float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))))
        val_loss = eg_mse + her_mse + 0.25 * bce
        physical = scaler.to_physical(val_raw)
        eg_rmse = rmse(val_batch.targets[:, 0], physical[:, 0])
        her_r = pearson(val_batch.targets[:, 1], physical[:, 1])
        return val_loss, eg_rmse, her_r


def train_ensemble(
    configs: Sequence[TrainConfig],
    featurizer: Featurizer,
    train_materials: Sequence[Material],
    val_materials: Sequence[Material],
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 128,
    gnn_dim: int = 32,
    verbose: bool = True,
) -> list[TrainingResult]:
    """Train one model per config (different seeds) sharing the featurizer."""
    if not configs:
        raise ValueError("at least one training config is required")
    results: list[TrainingResult] = []
    ctx_sizes = [featurizer.vocab_size(field) for field in featurizer.ctx_vocab]
    from .data.elements import ELEMENT_IDS

    element_vocab = max(ELEMENT_IDS.values()) + 1
    for i, config in enumerate(configs):
        model = GraphAugmentedTransformer(
            element_vocab_size=element_vocab,
            ctx_sizes=ctx_sizes,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            gnn_dim=gnn_dim,
            seed=config.seed,
        )
        if verbose:
            print(
                f"\n[ensemble {i + 1}/{len(configs)}] seed={config.seed} "
                f"params={model.num_params():,}",
                flush=True,
            )
        trainer = Trainer(config)
        results.append(trainer.train(model, featurizer, train_materials, val_materials))
    return results
