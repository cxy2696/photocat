"""Evaluation metrics, bootstrap confidence intervals and conformal intervals."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    if denominator < 1e-12:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator)


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    if y_true.size < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # Average ties.
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = np.mean(ranks[order[i:j]])
        i = j
    return ranks


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    if y_true.size < 2:
        return 0.0
    return pearson(_rankdata(y_true), _rankdata(y_pred))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(np.asarray(y_true).size),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "spearman": spearman(y_true, y_pred),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a paired statistic."""
    y_true, y_pred = _clean(y_true, y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 5:
        value = statistic(y_true, y_pred)
        return value, value
    values = np.empty(n_boot)
    for i in range(n_boot):
        index = rng.integers(0, n, n)
        values[i] = statistic(y_true[index], y_pred[index])
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))


def conformal_quantile(residuals: np.ndarray, coverage: float = 0.9) -> float:
    """Split-conformal quantile with the finite-sample correction."""
    residuals = np.abs(np.asarray(residuals, dtype=np.float64))
    residuals = residuals[np.isfinite(residuals)]
    n = len(residuals)
    if n == 0:
        return float("nan")
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(residuals, level, method="higher"))


def conformal_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    y_true, y_pred = _clean(y_true, y_pred)
    residual = np.abs(y_true - y_pred)
    inside = residual <= quantile
    return {
        "coverage": float(np.mean(inside)),
        "target_coverage": float("nan"),
        "mean_width": float(2.0 * quantile),
        "median_width": float(2.0 * quantile),
        "mean_residual": float(residual.mean()),
        "n": int(len(residual)),
    }


def auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney U)."""
    y_true = np.asarray(y_true).astype(bool)
    score = np.asarray(score, dtype=np.float64)
    positives = score[y_true]
    negatives = score[~y_true]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    # Probability a positive outranks a negative, with ties counted half.
    ranks = _rankdata(np.concatenate([positives, negatives]))
    rank_positives = ranks[: len(positives)].sum()
    n_pos = len(positives)
    n_neg = len(negatives)
    return float((rank_positives - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def top_k_recall(
    labels: np.ndarray, scores: np.ndarray, k: int, total_positives: int | None = None
) -> float:
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    k = min(k, len(scores))
    top = np.argsort(-scores)[:k]
    found = labels[top].sum()
    denominator = total_positives if total_positives is not None else labels.sum()
    return float(found / max(denominator, 1))


def enrichment_factor(
    labels: np.ndarray, scores: np.ndarray, k: int
) -> float:
    labels = np.asarray(labels).astype(bool)
    base_rate = labels.mean()
    if base_rate == 0:
        return 0.0
    top = np.argsort(-scores)[: min(k, len(scores))]
    return float(labels[top].mean() / base_rate)


def _clean(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]
