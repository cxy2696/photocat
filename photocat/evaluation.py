"""End-to-end evaluation: in-distribution, family holdout, anchors, baselines.

Reports the proposal's success criteria explicitly:

* bandgap RMSE < 0.3 eV,
* HER Pearson correlation > 0.8,
* novelty (median Tanimoto of shortlisted candidates) < 0.8.

Also reports 90% split-conformal coverage, deep-ensemble uncertainty/error
correlation, and a family-held-out stress test.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .data.dataset import Featurizer
from .data.synth import Material
from .metrics import (
    bootstrap_ci,
    conformal_quantile,
    mae,
    pearson,
    r2,
    regression_metrics,
    rmse,
    spearman,
)
from .models.baselines import KNNBaseline, MeanBaseline, RidgeBaseline
from .models.ensemble import EnsemblePrediction, ensemble_predict
from .training import TrainingResult


def _true_targets(materials: Sequence[Material]) -> np.ndarray:
    return np.asarray(
        [[material.eg, material.her, material.stability] for material in materials],
        dtype=np.float64,
    )


def _conformal_coverage(
    y_true: np.ndarray, y_pred: np.ndarray, quantile: float
) -> dict[str, float]:
    residual = np.abs(y_true - y_pred)
    return {
        "coverage": float(np.mean(residual <= quantile)),
        "mean_residual": float(residual.mean()),
        "width": float(2.0 * quantile),
        "n": int(len(residual)),
    }


def evaluate_split(
    materials: Sequence[Material],
    prediction: EnsemblePrediction,
    conformal: dict[str, float] | None = None,
    bootstrap: bool = True,
) -> dict[str, object]:
    truth = _true_targets(materials)
    eg = regression_metrics(truth[:, 0], prediction.eg)
    her = regression_metrics(truth[:, 1], prediction.her)
    stability = regression_metrics(truth[:, 2], prediction.stability)
    result: dict[str, object] = {
        "n": len(materials),
        "bandgap": eg,
        "her": her,
        "stability": stability,
        "uncertainty": {
            "eg_std_error_spearman": spearman(prediction.eg_std, np.abs(truth[:, 0] - prediction.eg)),
            "her_std_log_error_spearman": spearman(
                prediction.her_std,
                np.abs(np.log1p(truth[:, 1]) - np.log1p(np.clip(prediction.her, 0.0, None))),
            ),
            "eg_std_mean": float(prediction.eg_std.mean()),
            "her_std_mean": float(prediction.her_std.mean()),
        },
    }
    if bootstrap:
        result["bandgap"]["rmse_ci90"] = bootstrap_ci(
            truth[:, 0], prediction.eg, rmse, n_boot=600, seed=11
        )
        result["her"]["pearson_ci90"] = bootstrap_ci(
            truth[:, 1], prediction.her, pearson, n_boot=600, seed=12
        )
    if conformal:
        result["conformal"] = {
            "bandgap": _conformal_coverage(
                truth[:, 0], prediction.eg, conformal["bandgap_quantile"]
            ),
            "her_log": _conformal_coverage(
                np.log1p(truth[:, 1]),
                np.log1p(np.clip(prediction.her, 0.0, None)),
                conformal["her_quantile"],
            ),
        }
    return result


def fit_conformal(
    results: Sequence[TrainingResult],
    featurizer: Featurizer,
    materials: Sequence[Material],
    coverage: float = 0.9,
) -> dict[str, float]:
    prediction = ensemble_predict(results, featurizer, materials)
    truth = _true_targets(materials)
    eg_residual = np.abs(truth[:, 0] - prediction.eg)
    her_residual = np.abs(np.log1p(truth[:, 1]) - np.log1p(np.clip(prediction.her, 0.0, None)))
    return {
        "coverage": float(coverage),
        "bandgap_quantile": conformal_quantile(eg_residual, coverage),
        "her_quantile": conformal_quantile(her_residual, coverage),
        "n": int(len(materials)),
    }


def evaluate_baselines(
    train_materials: Sequence[Material],
    test_materials: Sequence[Material],
    ood_materials: Sequence[Material],
    featurizer: Featurizer,
) -> dict[str, object]:
    baselines: dict[str, object] = {}
    truth_test = _true_targets(test_materials)
    truth_ood = _true_targets(ood_materials)
    for name, baseline in (
        ("ridge", RidgeBaseline()),
        ("knn", KNNBaseline(k=12)),
        ("mean", MeanBaseline()),
    ):
        baseline.fit(train_materials, featurizer)
        test_pred = baseline.predict(test_materials, featurizer)
        ood_pred = baseline.predict(ood_materials, featurizer)
        baselines[name] = {
            "test": {
                "bandgap_rmse": rmse(truth_test[:, 0], test_pred.eg),
                "bandgap_pearson": pearson(truth_test[:, 0], test_pred.eg),
                "her_pearson": pearson(truth_test[:, 1], test_pred.her),
                "her_rmse": rmse(truth_test[:, 1], test_pred.her),
            },
            "ood": {
                "bandgap_rmse": rmse(truth_ood[:, 0], ood_pred.eg),
                "her_pearson": pearson(truth_ood[:, 1], ood_pred.her),
            },
        }
    return baselines


def evaluate_run(
    results: Sequence[TrainingResult],
    featurizer: Featurizer,
    splits: dict[str, Sequence[Material]],
    conformal: dict[str, float] | None = None,
    include_baselines: bool = True,
) -> dict[str, object]:
    predictions: dict[str, EnsemblePrediction] = {}
    evaluations: dict[str, object] = {}
    for name, materials in splits.items():
        if not materials:
            continue
        prediction = ensemble_predict(results, featurizer, materials)
        predictions[name] = prediction
        evaluations[name] = evaluate_split(
            materials, prediction, conformal=conformal, bootstrap=name in ("test", "ood")
        )

    if include_baselines and splits.get("train") and splits.get("test"):
        evaluations["baselines"] = evaluate_baselines(
            splits["train"], splits["test"], splits.get("ood", []), featurizer
        )

    test = evaluations.get("test", {})
    ood = evaluations.get("ood", {})
    criteria: dict[str, object] = {
        "bandgap_rmse_below_0.3_ev": bool(
            test.get("bandgap", {}).get("rmse", 99.0) < 0.3  # type: ignore[union-attr]
        ),
        "her_pearson_above_0.8": bool(
            test.get("her", {}).get("pearson", 0.0) > 0.8  # type: ignore[union-attr]
        ),
        "ood_bandgap_rmse": ood.get("bandgap", {}).get("rmse"),  # type: ignore[union-attr]
        "ood_her_pearson": ood.get("her", {}).get("pearson"),  # type: ignore[union-attr]
    }
    evaluations["success_criteria"] = criteria
    evaluations["predictions"] = _prediction_rows(splits, predictions)
    return evaluations


def _prediction_rows(
    splits: dict[str, Sequence[Material]], predictions: dict[str, EnsemblePrediction]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, materials in splits.items():
        prediction = predictions.get(split)
        if prediction is None:
            continue
        for i, material in enumerate(materials):
            rows.append(
                {
                    "split": split,
                    "id": material.id,
                    "label": material.label,
                    "formula": material.formula,
                    "family": material.family,
                    "true_eg": round(material.eg, 4),
                    "pred_eg": round(float(prediction.eg[i]), 4),
                    "eg_std": round(float(prediction.eg_std[i]), 4),
                    "true_her": round(material.her, 4),
                    "pred_her": round(float(prediction.her[i]), 4),
                    "her_std": round(float(prediction.her_std[i]), 4),
                    "true_stability": round(material.stability, 4),
                    "pred_stability": round(float(prediction.stability[i]), 4),
                }
            )
    return rows


def summarize_predictions(
    materials: Sequence[Material],
    prediction: EnsemblePrediction,
) -> dict[str, float]:
    """Small diagnostic block used by the report."""
    truth = _true_targets(materials)
    return {
        "bandgap_mae": mae(truth[:, 0], prediction.eg),
        "bandgap_r2": r2(truth[:, 0], prediction.eg),
        "her_spearman": spearman(truth[:, 1], prediction.her),
        "stability_mae": mae(truth[:, 2], prediction.stability),
    }
