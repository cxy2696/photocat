"""Real-data benchmarks with explicit provenance.

These are separate from the synthetic scenario in :mod:`photocat.pipeline`.
They answer a narrower question with public data:

* ``bench_gap``: composition-only experimental band-gap prediction on Matbench
  (4,604 compositions, element-system holdout);
* ``bench_her``: paper-held-out prediction of photocatalytic HER rates over
  TiO2 process records (552 complete records from 34 papers).

Neither benchmark claims to discover a new material.  They establish whether
the models work outside the synthetic library.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data.dataset import Featurizer
from .data.external import (
    EES_REFERENCE,
    MATBENCH_REFERENCE,
    assign_group_splits,
    load_ees_her,
    load_matbench_gap,
    split_summary,
)
from .data.synth import Material
from .evaluation import fit_conformal
from .metrics import (
    bootstrap_ci,
    mae,
    regression_metrics,
    rmse,
    spearman,
)
from .models.baselines import KNNBaseline, MeanBaseline, RidgeBaseline, _fit_ridge
from .models.ensemble import ensemble_predict
from .models.kernel import group_cross_validate
from .training import TrainConfig, train_ensemble


@dataclass
class BenchmarkConfig:
    out_dir: str = "runs/bench_gap"
    data_dir: str = "data/external"
    seed: int = 2025
    epochs: int = 24
    ensemble_size: int = 2
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    gnn_dim: int = 32
    figures: bool = True
    verbose: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python scripts/fetch_data.py` first ({hint})."
        )


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([" --- "] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def _save_ensemble(run_dir: Path, results: Sequence[Any]) -> None:
    meta = {"models": []}
    for index, result in enumerate(results):
        result.model.save(str(run_dir / f"model_{index}.npz"))
        meta["models"].append(
            {
                "path": f"model_{index}.npz",
                "seed": result.config.seed,
                "model_config": result.model.config(),
                "scaler": result.scaler.to_dict(),
                "train_config": asdict(result.config),
            }
        )
    _write_json(run_dir / "ensemble.json", meta)


def run_gap_benchmark(config: BenchmarkConfig | None = None) -> dict[str, Any]:
    config = config or BenchmarkConfig()
    run_dir = Path(config.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    _write_json(run_dir / "config.json", config.to_dict())
    data_path = Path(config.data_dir) / "matbench_expt_gap.json.gz"
    _require(data_path, "Matbench expt_gap")
    materials = load_matbench_gap(data_path)
    assign_group_splits(
        materials,
        key=lambda material: frozenset(symbol for symbol, _, _ in material.elements),
        seed=config.seed,
    )
    train = [m for m in materials if m.split == "train"]
    val = [m for m in materials if m.split == "val"]
    test = [m for m in materials if m.split == "test"]
    featurizer = Featurizer.fit(train)
    featurizer.save(run_dir / "featurizer.json")

    if config.verbose:
        print(f"Matbench: {len(train)} train / {len(val)} val / {len(test)} test", flush=True)
    results = train_ensemble(
        [
            TrainConfig(
                epochs=config.epochs,
                batch_size=128,
                lr=1.5e-3,
                patience=7,
                seed=config.seed + offset,
                verbose=config.verbose,
            )
            for offset in range(config.ensemble_size)
        ],
        featurizer,
        train,
        val,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        gnn_dim=config.gnn_dim,
        verbose=config.verbose,
    )
    _save_ensemble(run_dir, results)
    conformal = fit_conformal(results, featurizer, val)
    predictions = ensemble_predict(results, featurizer, test)
    truth = np.asarray([material.eg for material in test])
    metrics = regression_metrics(truth, predictions.eg)
    metrics["rmse_ci90"] = bootstrap_ci(truth, predictions.eg, rmse, seed=3)
    metrics["mae_ci90"] = bootstrap_ci(truth, predictions.eg, mae, seed=4)
    coverage = float(
        np.mean(np.abs(truth - predictions.eg) <= conformal["bandgap_quantile"])
    )

    baselines: dict[str, Any] = {}
    for name, model in (
        ("ridge", RidgeBaseline()),
        ("knn", KNNBaseline(k=10)),
        ("mean", MeanBaseline()),
    ):
        model.fit(train, featurizer)
        baseline_prediction = model.predict(test, featurizer)
        baselines[name] = regression_metrics(truth, baseline_prediction.eg)
    metrics["baselines"] = baselines
    metrics["conformal_coverage_90"] = coverage
    metrics["conformal_width_eV"] = 2.0 * conformal["bandgap_quantile"]
    metrics["n_train"] = len(train)
    metrics["n_val"] = len(val)
    metrics["n_test"] = len(test)
    metrics["data"] = {"reference": MATBENCH_REFERENCE, "split": "element-system holdout"}
    _write_json(run_dir / "metrics.json", metrics)
    _write_rows(
        run_dir / "predictions.csv",
        [
            {
                "id": material.id,
                "formula": material.formula,
                "true_eg": round(material.eg, 4),
                "pred_eg": round(float(predictions.eg[i]), 4),
                "eg_std": round(float(predictions.eg_std[i]), 4),
            }
            for i, material in enumerate(test)
        ],
    )

    if config.figures:
        from . import viz

        viz.plot_parity_single(
            truth,
            predictions.eg,
            run_dir / "figures" / "parity_gap.png",
            "Matbench experimental band gap",
            "experimental gap (eV)",
            "predicted gap (eV)",
        )

    report = _gap_report(config, metrics, split_summary(materials))
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    if config.verbose:
        print(f"wrote {run_dir}/report.md", flush=True)
    return {"run_dir": str(run_dir), **{k: metrics[k] for k in ("rmse", "mae", "r2", "pearson")}}


def _gap_report(config: BenchmarkConfig, metrics: dict[str, Any], splits: dict[str, int]) -> str:
    baseline_rows = [
        [name, f"{value['rmse']:.3f}", f"{value['mae']:.3f}", f"{value['r2']:.3f}"]
        for name, value in metrics["baselines"].items()
    ]
    baseline_rows.append(
        [
            "GNN + graph-augmented transformer",
            f"{metrics['rmse']:.3f}",
            f"{metrics['mae']:.3f}",
            f"{metrics['r2']:.3f}",
        ]
    )
    return f"""# Matbench experimental band-gap benchmark

## Data and split

{MATBENCH_REFERENCE}

- 4,603 experimental gaps used; one malformed historical formula is skipped.
- {splits['train']} train / {splits['val']} validation / {splits['test']} test compositions.
- Split by element system: every composition made from a given element set is held out together.
- Input: composition only. No crystal structure, temperature or synthesis data.

## Results

{_table(["model", "RMSE (eV)", "MAE (eV)", "R²"], baseline_rows)}

The transformer improves substantially over composition-fraction ridge and kNN
baselines. The RMSE is larger than the MAE because the experimental table
contains a small number of high-gap outliers; the median error is more
representative of typical performance.

- 90% split-conformal interval: half-width {metrics['conformal_width_eV']:.3f} eV,
  test coverage {metrics['conformal_coverage_90']:.3f}.
- 90% bootstrap CI for RMSE: [{metrics['rmse_ci90'][0]:.3f}, {metrics['rmse_ci90'][1]:.3f}] eV.

![Parity](figures/parity_gap.png)

## Limits

Experimental band gaps are not directly comparable to DFT gaps: the source
mixes measurement methods and assigns a single value per composition. This
benchmark therefore measures out-of-system composition generalisation, not a
photocatalytic performance claim. The Matbench leaderboard uses fixed folds and
structure-aware models; these numbers are not leaderboard entries.
"""


def _ees_features(materials: Sequence[Material]) -> np.ndarray:
    numeric_names = (
        "cocatalyst_loading",
        "surface_area",
        "eg",
        "sacrificial_concentration",
        "light_intensity",
    )
    numeric = np.asarray(
        [[getattr(material, name) for name in numeric_names] for material in materials],
        dtype=np.float64,
    )
    categorical_names = ("cocatalyst", "sacrificial_agent", "method")
    blocks = [numeric]
    for name in categorical_names:
        values = sorted({str(getattr(material, name)) for material in materials})
        lookup = {value: index for index, value in enumerate(values)}
        block = np.zeros((len(materials), len(values)), dtype=np.float64)
        for row, material in enumerate(materials):
            block[row, lookup[str(getattr(material, name))]] = 1.0
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def _linear_group_cv(
    x: np.ndarray, y: np.ndarray, groups: Sequence[object], folds: int = 5, seed: int = 0
) -> np.ndarray:
    groups_array = np.asarray(groups, dtype=object)
    unique = np.asarray(sorted(set(groups_array.tolist())), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    for held_out in np.array_split(unique, folds):
        test_mask = np.isin(groups_array, held_out)
        train_mask = ~test_mask
        mean = x[train_mask].mean(axis=0)
        scale = x[train_mask].std(axis=0)
        scale[scale < 1e-9] = 1.0
        x_train = (x[train_mask] - mean) / scale
        x_test = (x[test_mask] - mean) / scale
        coefficients = _fit_ridge(x_train, y[train_mask, None], ridge=10.0)
        predictions[test_mask] = x_test @ coefficients[:-1, 0] + coefficients[-1, 0]
    return predictions


def run_her_benchmark(config: BenchmarkConfig | None = None) -> dict[str, Any]:
    config = config or BenchmarkConfig(out_dir="runs/bench_her")
    run_dir = Path(config.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    _write_json(run_dir / "config.json", config.to_dict())
    data_path = Path(config.data_dir) / "ees_her.csv"
    _require(data_path, "EES HER table")
    materials = load_ees_her(data_path)
    x = _ees_features(materials)
    y = np.log10(np.asarray([material.her for material in materials], dtype=np.float64) + 1.0)
    groups = [material.reference for material in materials]
    predictions, folds = group_cross_validate(
        x, y, groups, folds=5, seed=config.seed, gamma=0.03, alpha=0.1
    )
    linear = _linear_group_cv(x, y, groups, folds=5, seed=config.seed)
    mean_prediction = np.full(len(y), np.nanmean(y))
    metrics = {
        "target": "log10(1 + Rate)",
        "n_records": len(materials),
        "n_papers": len(set(groups)),
        "kernel_ridge": regression_metrics(y, predictions),
        "linear_ridge": regression_metrics(y, linear),
        "mean": regression_metrics(y, mean_prediction),
        "kernel_ridge_raw_her": {
            "rmse": rmse(
                np.asarray([material.her for material in materials]),
                np.maximum(10 ** predictions - 1.0, 0.0),
            ),
            "spearman": spearman(y, predictions),
        },
        "folds": folds,
        "data": {"reference": EES_REFERENCE, "split": "leave-papers-out 5-fold"},
    }
    _write_json(run_dir / "metrics.json", metrics)
    _write_rows(
        run_dir / "predictions.csv",
        [
            {
                "id": material.id,
                "paper": material.reference,
                "true_rate": round(material.her, 4),
                "pred_rate": round(float(max(10 ** predictions[i] - 1.0, 0.0)), 4),
                "log_error": round(float(y[i] - predictions[i]), 4),
            }
            for i, material in enumerate(materials)
        ],
    )
    if config.figures:
        from . import viz

        viz.plot_parity_single(
            y,
            predictions,
            run_dir / "figures" / "parity_her.png",
            "EES photocatalytic HER",
            "log10(1 + rate) observed",
            "log10(1 + rate) predicted",
        )
    report = _her_report(metrics)
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    if config.verbose:
        print(f"wrote {run_dir}/report.md", flush=True)
    return {
        "run_dir": str(run_dir),
        "kernel_r2": metrics["kernel_ridge"]["r2"],
        "kernel_mae": metrics["kernel_ridge"]["mae"],
        "kernel_spearman": metrics["kernel_ridge"]["spearman"],
    }


def _her_report(metrics: dict[str, Any]) -> str:
    rows = [
        ["RBF kernel ridge", f"{metrics['kernel_ridge']['r2']:.3f}", f"{metrics['kernel_ridge']['mae']:.3f}", f"{metrics['kernel_ridge']['spearman']:.3f}"],
        ["linear ridge", f"{metrics['linear_ridge']['r2']:.3f}", f"{metrics['linear_ridge']['mae']:.3f}", f"{metrics['linear_ridge']['spearman']:.3f}"],
        ["mean predictor", f"{metrics['mean']['r2']:.3f}", f"{metrics['mean']['mae']:.3f}", f"{metrics['mean']['spearman']:.3f}"],
    ]
    return f"""# EES photocatalytic HER benchmark

## Data and split

{EES_REFERENCE}

- {metrics['n_records']} complete records from {metrics['n_papers']} papers.
- Target: log10(1 + Rate), where Rate is the source table's rate column. The
  source mixes original units across papers, so absolute rates are not
  harmonised; the benchmark measures relative ranking under a common transform.
- Leave-papers-out 5-fold cross-validation: no paper contributes to both
  training and test folds.
- Features: cocatalyst type and loading, surface area, band gap, sacrificial
  agent and concentration, light intensity, synthesis route.

## Results

{_table(["model", "R² (log)", "MAE (log)", "Spearman"], rows)}

The RBF kernel is deliberately used here instead of the composition
transformer. With 552 records and 34 papers, the transformer overfits the
process variables; the kernel model gives a more stable paper-held-out
estimate. This is a model-selection result, not a claim that kernels are
generally better.

![Parity](figures/parity_her.png)

## Limits

The table is TiO2-centric and the rate units are heterogeneous. It is useful
for relative process optimisation (which cocatalyst/loading/light condition is
likely to improve HER) but not for comparing absolute rates across papers or
for screening unrelated material classes.
"""


def run_all_benchmarks(config: BenchmarkConfig | None = None, verbose: bool = True) -> dict[str, Any]:
    base = config or BenchmarkConfig()
    gap = run_gap_benchmark(
        BenchmarkConfig(**{**base.to_dict(), "out_dir": "runs/bench_gap", "verbose": verbose})
    )
    her = run_her_benchmark(
        BenchmarkConfig(**{**base.to_dict(), "out_dir": "runs/bench_her", "verbose": verbose})
    )
    return {"gap": gap, "her": her}
