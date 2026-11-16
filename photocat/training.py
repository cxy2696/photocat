"""Training loop, target scaling, batching and ensemble inference."""

from __future__ import annotations

import math
import time
import warnings
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
    """Standardisation for available targets; missing targets stay missing.

    A target column with no observations is marked ``observed=False`` and is
    returned as NaN at inference time.  This is what lets the same trainer use
    Matbench (bandgap only), EES (HER only) and the synthetic scenario
    (all three) without inventing labels.
    """

    mean: np.ndarray
    std: np.ndarray
    observed: np.ndarray = field(default_factory=lambda: np.ones(3, dtype=bool))

    @classmethod
    def fit(cls, targets: np.ndarray) -> "TargetScaler":
        targets = np.asarray(targets, dtype=np.float64)
        observed = np.isfinite(targets).sum(axis=0) > 0
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(targets, axis=0)
            std = np.nanstd(targets, axis=0)
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=1.0)
        std[std < 1e-8] = 1.0
        mean[2] = 0.0
        std[2] = 1.0
        return cls(mean=mean, std=std, observed=observed)

    def scale(self, targets: np.ndarray) -> np.ndarray:
        return (np.asarray(targets, dtype=np.float64) - self.mean) / self.std

    def to_physical(self, predictions: np.ndarray) -> np.ndarray:
        predictions = np.asarray(predictions, dtype=np.float64)
        eg = predictions[:, 0] * self.std[0] + self.mean[0]
        her = np.expm1(predictions[:, 1] * self.std[1] + self.mean[1])
        stability = 1.0 / (1.0 + np.exp(-predictions[:, 2]))
        physical = np.stack([eg, her, stability], axis=-1)
        physical[:, ~self.observed] = np.nan
        return physical

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "observed": self.observed.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TargetScaler":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
            observed=np.asarray(payload.get("observed", [True, True, True]), dtype=bool),
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
    training: bool = False,
) -> dict[str, object]:
    """Run the model over a batch in chunks; returns raw (model-unit) outputs."""
    raw_chunks: list[np.ndarray] = []
    attention_chunks: list[list[np.ndarray]] | None = [] if return_attention else None
    for start in range(0, len(batch), batch_size):
        chunk = batch.take(np.arange(start, min(start + batch_size, len(batch))))
        output = model.forward(chunk, return_attention=return_attention, training=training)
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
    training: bool = False,
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
        target_masks = np.isfinite(train_batch.targets)
        safe_scaled = np.nan_to_num(scaled_targets, nan=0.0)
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
                output = model.forward(chunk, training=True)
                masks = target_masks[index]
                loss_terms = []
                for task, (head, weight) in enumerate(
                    zip((output.eg, output.her), config.loss_weights[:2])
                ):
                    task_weights = weights * masks[:, task]
                    if task_weights.sum() > 0:
                        loss_terms.append(
                            weight * mse_loss(head, safe_scaled[index, task], task_weights)
                        )
                stability_weights = weights * masks[:, 2]
                if stability_weights.sum() > 0:
                    loss_terms.append(
                        config.loss_weights[2]
                        * bce_with_logits(
                            output.stability, safe_raw[index, 2], stability_weights
                        )
                    )
                if not loss_terms:
                    continue
                loss = loss_terms[0]
                for term in loss_terms[1:]:
                    loss = loss + term
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
                parts = [f"epoch {epoch:3d}", f"train {train_loss:.4f}", f"val {val_loss:.4f}"]
                if np.isfinite(val_eg_rmse):
                    parts.append(f"Eg RMSE {val_eg_rmse:.3f} eV")
                if np.isfinite(val_her_pearson):
                    parts.append(f"HER r {val_her_pearson:.3f}")
                print(" | ".join(parts), flush=True)
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
        masks = np.isfinite(val_batch.targets)
        losses: list[float] = []
        for task in range(2):
            if masks[:, task].any():
                losses.append(
                    float(np.mean((val_raw[masks[:, task], task] - scaled_val[masks[:, task], task]) ** 2))
                )
        if masks[:, 2].any():
            z = val_raw[masks[:, 2], 2]
            y = val_batch.targets[masks[:, 2], 2]
            losses.append(
                0.25
                * float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))))
            )
        val_loss = float(sum(losses)) if losses else float("inf")
        physical = scaler.to_physical(val_raw)
        eg_rmse = (
            rmse(val_batch.targets[masks[:, 0], 0], physical[masks[:, 0], 0])
            if masks[:, 0].any()
            else float("nan")
        )
        her_r = (
            pearson(val_batch.targets[masks[:, 1], 1], physical[masks[:, 1], 1])
            if masks[:, 1].any()
            else float("nan")
        )
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
    dropout: float = 0.1,
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
            dropout=dropout,
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
