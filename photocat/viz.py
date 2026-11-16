"""Matplotlib figures for the report (optional dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .metrics import pearson, rmse


def _plt():
    import os
    from pathlib import Path

    cache = Path(os.environ.get("MPLCONFIGDIR", ".mplconfig"))
    try:
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    except OSError:
        pass
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 170,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 11,
            "legend.frameon": False,
        }
    )
    return plt


def _save(figure, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return str(path)


def plot_training_curves(histories: Sequence[list[dict[str, float]]], path: str | Path) -> str:
    plt = _plt()
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.7))
    for i, history in enumerate(histories):
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], alpha=0.35, lw=1.0)
        axes[0].plot(epochs, [row["val_loss"] for row in history], label=f"model {i}", lw=1.4)
        axes[1].plot(epochs, [row["val_eg_rmse"] for row in history], label=f"model {i}")
        axes[2].plot(epochs, [row["val_her_pearson"] for row in history], label=f"model {i}")
    axes[0].set(title="Loss", xlabel="epoch", ylabel="weighted loss")
    axes[1].set(title="Bandgap RMSE", xlabel="epoch", ylabel="eV")
    axes[1].axhline(0.3, color="crimson", ls="--", lw=1.2, label="target 0.3 eV")
    axes[2].set(title="HER Pearson", xlabel="epoch", ylabel="r")
    axes[2].axhline(0.8, color="crimson", ls="--", lw=1.2, label="target 0.8")
    axes[2].set_ylim(min(0.0, min(min(r["val_her_pearson"] for r in h) for h in histories) - 0.05), 1.0)
    for axis in axes:
        axis.legend(fontsize=8)
    return _save(figure, path)


def plot_parity(
    test_truth: np.ndarray,
    test_pred: np.ndarray,
    ood_truth: np.ndarray,
    ood_pred: np.ndarray,
    path: str | Path,
    title: str = "Bandgap parity",
    unit: str = "eV",
    log_y: bool = False,
) -> str:
    plt = _plt()
    figure, axis = plt.subplots(figsize=(4.6, 4.4))
    low = float(min(test_truth.min(), ood_truth.min()))
    high = float(max(test_truth.max(), ood_truth.max()))
    if log_y:
        low, high = max(low, 0.01), max(high, 0.01)
    axis.scatter(test_truth, test_pred, s=14, alpha=0.55, c="#1f77b4", label="in-distribution test")
    axis.scatter(ood_truth, ood_pred, s=14, alpha=0.55, c="#ff7f0e", label="family holdout")
    axis.plot([low, high], [low, high], color="black", lw=1.0, ls="--")
    if log_y:
        axis.set_xscale("log")
        axis.set_yscale("log")
    axis.set_xlim(low * 0.95 if log_y else low - 0.1, high * 1.05 if log_y else high + 0.1)
    axis.set_ylim(low * 0.95 if log_y else low - 0.1, high * 1.05 if log_y else high + 0.1)
    axis.set_title(title)
    axis.set_xlabel(f"true {unit}")
    axis.set_ylabel(f"predicted {unit}")
    axis.text(
        0.04,
        0.96,
        f"test RMSE {rmse(test_truth, test_pred):.3f}\ntest r {pearson(test_truth, test_pred):.3f}\n"
        f"OOD RMSE {rmse(ood_truth, ood_pred):.3f}",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    axis.legend(fontsize=8, loc="lower right")
    return _save(figure, path)


def plot_parity_single(
    truth: np.ndarray,
    prediction: np.ndarray,
    path: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
    log_axes: bool = False,
) -> str:
    plt = _plt()
    figure, axis = plt.subplots(figsize=(4.6, 4.4))
    low = float(min(truth.min(), prediction.min()))
    high = float(max(truth.max(), prediction.max()))
    axis.scatter(truth, prediction, s=14, alpha=0.55, c="#31688e")
    axis.plot([low, high], [low, high], color="black", lw=1.0, ls="--")
    if log_axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.text(
        0.04,
        0.96,
        f"n = {len(truth)}\nRMSE = {rmse(truth, prediction):.3f}\n"
        f"MAE = {np.mean(np.abs(truth - prediction)):.3f}",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    return _save(figure, path)


def plot_screening(
    eg: np.ndarray,
    her: np.ndarray,
    selected: Sequence[tuple[float, float, str]],
    path: str | Path,
) -> str:
    plt = _plt()
    figure, axis = plt.subplots(figsize=(6.2, 4.3))
    axis.scatter(eg, np.log10(her + 1.0), s=9, alpha=0.3, c="#9aa7b4", label="pool")
    axis.axvspan(1.23, 3.0, color="#2ca02c", alpha=0.08, label="bandgap window")
    axis.axhline(np.log10(7.0), color="#2ca02c", ls="--", lw=1.0, label="HER = 6")
    for e, h, _ in selected:
        axis.scatter([e], [np.log10(h + 1.0)], s=70, marker="*", c="crimson", zorder=5)
    if selected:
        top_names = "\n".join(
            f"{rank + 1}. {label}" for rank, (_, _, label) in enumerate(selected[:3])
        )
        axis.text(
            0.985,
            0.03,
            "shortlist:\n" + top_names,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2.0},
        )
    axis.set(xlabel="predicted bandgap (eV)", ylabel="log10(HER + 1)", title="Screening funnel")
    axis.legend(fontsize=8, loc="upper right")
    return _save(figure, path)


def plot_bias_audit(
    train_counts: dict[str, int], test_counts: dict[str, int], ood_counts: dict[str, int], path: str | Path
) -> str:
    plt = _plt()
    families = sorted(train_counts)
    x = np.arange(len(families))
    width = 0.26
    figure, axis = plt.subplots(figsize=(8.4, 3.6))
    axis.bar(x - width, [train_counts[f] for f in families], width, label="train")
    axis.bar(x, [test_counts.get(f, 0) for f in families], width, label="test")
    axis.bar(x + width, [ood_counts.get(f, 0) for f in families], width, label="family holdout")
    axis.set_xticks(x, [family.replace("_", " ") for family in families], rotation=25, ha="right")
    axis.set_ylabel("materials")
    axis.set_title("Crystal-family coverage and bias audit")
    axis.legend(fontsize=8)
    return _save(figure, path)


def plot_active_learning(curve: Sequence[dict[str, float]], path: str | Path) -> str:
    plt = _plt()
    figure, axis = plt.subplots(figsize=(5.2, 3.7))
    budgets = [row["budget"] for row in curve]
    axis.plot(budgets, [row["active_recall"] for row in curve], marker="o", label="KG-guided acquisition")
    axis.plot(budgets, [row["random_recall"] for row in curve], marker="s", label="random sampling")
    axis.set(xlabel="new labelled experiments", ylabel="recall of top performers", ylim=(-0.02, 1.05))
    axis.set_title("Simulated experimental iterations")
    axis.legend(fontsize=8)
    return _save(figure, path)


def plot_attention(tokens: Sequence[str], importance: np.ndarray, path: str | Path) -> str:
    plt = _plt()
    values = np.asarray(importance, dtype=float)
    keep = [i for i, token in enumerate(tokens) if token not in ("[CLS]", "[PAD]")]
    order = np.asarray(keep)[np.argsort(-values[keep])[:10]][::-1]
    figure, axis = plt.subplots(figsize=(5.6, 4.2))
    axis.barh([tokens[i] for i in order], values[order], color="#4c78a8")
    axis.set_xlabel("CLS attention rollout weight")
    axis.set_title("Which tokens drive the prediction?")
    return _save(figure, path)


def plot_uncertainty(
    eg_std: np.ndarray, absolute_error: np.ndarray, path: str | Path
) -> str:
    plt = _plt()
    figure, axis = plt.subplots(figsize=(4.5, 3.7))
    axis.scatter(eg_std, absolute_error, s=14, alpha=0.45, c="#4c78a8")
    axis.set(xlabel="ensemble std of Eg (eV)", ylabel="|error| (eV)", title="Epistemic uncertainty vs error")
    return _save(figure, path)


def plot_shapley(attribution: dict[str, float], label: str, path: str | Path) -> str:
    plt = _plt()
    items = sorted(attribution.items(), key=lambda item: abs(item[1]), reverse=True)[:10][::-1]
    figure, axis = plt.subplots(figsize=(4.8, 3.4))
    axis.barh([item[0] for item in items], [item[1] for item in items], color="#54a24b")
    axis.axvline(0.0, color="black", lw=0.8)
    axis.set_xlabel("Shapley value for predicted HER (mmol g-1 h-1)")
    axis.set_title(f"Element contributions: {label}")
    return _save(figure, path)
