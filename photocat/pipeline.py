"""End-to-end pipeline: data -> KG -> training -> evaluation -> screening -> report."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .active import ActiveLearningConfig, simulate_active_learning
from .data.dataset import Featurizer
from .data.fingerprint import fingerprint_matrix, max_similarity
from .data.synth import (
    FAMILIES,
    DatasetBundle,
    generate_dataset,
    save_materials,
    summary,
)
from .evaluation import evaluate_run, fit_conformal
from .interpret import (
    attention_rollout,
    element_shapley,
    permutation_importance,
)
from .kg.corpus import generate_corpus, save_corpus
from .kg.extraction import evaluate_extraction, extract, load_extractions
from .kg.link_prediction import train_link_predictor
from .kg.reasoning import explain, format_explanation, run_reasoning
from .kg.store import build_kg
from .models.ensemble import ensemble_predict
from .models.transformer import GraphAugmentedTransformer
from .ranking import RankConfig, screen_candidates
from .report import generate_report
from .training import (
    TargetScaler,
    TrainConfig,
    TrainingHistory,
    TrainingResult,
    predict_materials,
    train_ensemble,
)

@dataclass
class PipelineConfig:
    name: str = "demo"
    out_dir: str = "runs/demo"
    n_materials: int = 12000
    seed: int = 2025
    holdout_families: tuple[str, ...] = ("oxynitride", "sulfide")
    epochs: int = 30
    batch_size: int = 128
    ensemble_size: int = 3
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    gnn_dim: int = 32
    mitigation: str = "none"
    top_k: int = 50
    n_corpus: int = 600
    link_epochs: int = 12
    run_link_prediction: bool = True
    al_initial: int = 150
    al_batch: int = 100
    al_rounds: int = 12
    al_bootstrap: int = 5
    figures: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _counts(materials: list) -> dict[str, int]:
    return {family: sum(material.family == family for material in materials) for family in FAMILIES}


def _plot(run_dir: Path, name: str, function: Any, *args: Any, **kwargs: Any) -> bool:
    try:
        function(*args, path=run_dir / "figures" / name, **kwargs)
        return True
    except ImportError:
        return False
    except Exception as error:  # pragma: no cover - plotting is best effort
        print(f"[figures] {name} failed: {error}")
        return False


def run_pipeline(config: PipelineConfig | None = None, verbose: bool = True) -> dict[str, Any]:
    config = config or PipelineConfig()
    run_dir = Path(config.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", config.to_dict())
    started = time.time()

    # ------------------------------------------------------------------ 1. data
    if verbose:
        print("[1/9] generating physics-informed library ...", flush=True)
    bundle: DatasetBundle = generate_dataset(
        n=config.n_materials,
        seed=config.seed,
        holdout_families=config.holdout_families,
    )
    save_materials(run_dir / "materials.csv", bundle.materials)
    save_materials(run_dir / "anchors.csv", bundle.anchors)
    corpus = generate_corpus(
        bundle.materials, bundle.anchors, n=config.n_corpus, seed=config.seed
    )
    save_corpus(run_dir / "corpus.jsonl", corpus)
    data_summary = summary(bundle.materials)
    data_summary.update(
        {
            "train": len(bundle.train),
            "val": len(bundle.val),
            "test": len(bundle.test),
            "ood": len(bundle.ood),
            "holdout_families": list(config.holdout_families),
            "anchors": len(bundle.anchors),
        }
    )

    # -------------------------------------------------------------------- 2. KG
    if verbose:
        print("[2/9] building knowledge graph, extracting relations, training DistMult ...", flush=True)
    kg_path = run_dir / "kg.sqlite"
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(kg_path) + suffix)
        if candidate.exists():
            candidate.unlink()
    store = build_kg(kg_path, bundle.materials, bundle.anchors)
    extraction_predictions = [extract(record.text) for record in corpus]
    extraction_metrics = evaluate_extraction(corpus, extraction_predictions)
    material_by_label: dict[str, Any] = {}
    for material in list(bundle.materials) + list(bundle.anchors):
        material_by_label.setdefault(material.label, material)
    extraction_metrics["linking"] = load_extractions(store, corpus, material_by_label)

    link_result = None
    if config.run_link_prediction and config.link_epochs > 0:
        link_result = train_link_predictor(
            store,
            epochs=config.link_epochs,
            dim=32,
            lr=0.02,
            seed=config.seed + 7,
            full_vocab_eval=200,
        )

    train_fingerprints = fingerprint_matrix(bundle.train)
    all_materials = list(bundle.materials) + list(bundle.anchors)
    all_fingerprints = fingerprint_matrix(all_materials)
    similarity, _ = max_similarity(all_fingerprints, train_fingerprints)
    similarity_map = {
        material.id: float(value) for material, value in zip(all_materials, similarity)
    }
    rank_config = RankConfig(top_k=config.top_k)
    reasoning = run_reasoning(store, rank_config, similarity_map, max_heterojunctions=20)
    explanation_payload: dict[str, Any] = {}
    if reasoning.candidates:
        top_candidate_id = str(reasoning.candidates[0]["material_id"])
        explanation = explain(store, top_candidate_id)
        explanation_payload = {
            "material_id": top_candidate_id,
            "text": format_explanation(explanation),
            "summary": explanation["summary"],
        }
    kg_metrics = {
        "stats": store.stats(),
        "extraction": extraction_metrics,
        "link_prediction": link_result.metrics if link_result else {},
        "link_examples": link_result.examples if link_result else [],
        "reasoning": reasoning.stats,
        "heterojunctions": reasoning.heterojunctions,
        "explanation": explanation_payload,
    }
    _write_json(run_dir / "kg_metrics.json", kg_metrics)

    # ------------------------------------------------------------- 3. features
    if verbose:
        print("[3/9] fitting featurizer ...", flush=True)
    featurizer = Featurizer.fit(bundle.train)
    featurizer.save(run_dir / "featurizer.json")

    # -------------------------------------------------------------- 4. training
    if verbose:
        print(
            f"[4/9] training {config.ensemble_size}-model ensemble "
            f"({config.epochs} epochs each) ...",
            flush=True,
        )
    train_configs = [
        TrainConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            seed=config.seed + i,
            mitigation=config.mitigation,
        )
        for i in range(config.ensemble_size)
    ]
    results = train_ensemble(
        train_configs,
        featurizer,
        bundle.train,
        bundle.val,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        gnn_dim=config.gnn_dim,
        verbose=verbose,
    )

    ensemble_meta: dict[str, Any] = {"models": []}
    history: dict[str, Any] = {"models": []}
    for i, result in enumerate(results):
        model_path = f"model_{i}.npz"
        result.model.save(str(run_dir / model_path))
        ensemble_meta["models"].append(
            {
                "path": model_path,
                "seed": result.config.seed,
                "model_config": result.model.config(),
                "scaler": result.scaler.to_dict(),
                "train_config": asdict(result.config),
            }
        )
        history["models"].append(
            {
                "seed": result.config.seed,
                "params": result.model.num_params(),
                "best_epoch": result.history.best_epoch,
                "seconds": round(result.history.seconds, 2),
                "best_val_loss": result.history.best_val_loss,
                "epochs": result.history.epochs,
            }
        )
    _write_json(run_dir / "ensemble.json", ensemble_meta)
    _write_json(run_dir / "history.json", history)

    # ------------------------------------------------------------ 5. evaluation
    if verbose:
        print("[5/9] evaluating in-distribution, family holdout and anchors ...", flush=True)
    conformal = fit_conformal(results, featurizer, bundle.val)
    splits = {
        "train": bundle.train,
        "val": bundle.val,
        "test": bundle.test,
        "ood": bundle.ood,
        "anchors": bundle.anchors,
    }
    evaluations = evaluate_run(results, featurizer, splits, conformal=conformal)
    prediction_rows = evaluations.pop("predictions")
    _write_rows(run_dir / "predictions.csv", prediction_rows)

    # -------------------------------------------------------------- 6. ranking
    if verbose:
        print("[6/9] screening and ranking candidates ...", flush=True)
    pool = list(bundle.test) + list(bundle.ood)
    pool_prediction = ensemble_predict(results, featurizer, pool)
    candidates, funnel = screen_candidates(
        pool, pool_prediction.as_dict(), bundle.train, rank_config
    )
    _write_rows(run_dir / "rankings.csv", [candidate.to_row() for candidate in candidates])

    novelty_median = funnel.get("similarity_median", float("nan"))
    success_criteria = evaluations.get("success_criteria", {})
    success_criteria["novelty_below_0.8"] = bool(
        np.isfinite(novelty_median) and novelty_median < rank_config.novelty_max
    )

    # ------------------------------------------------------- 7. active learning
    if verbose:
        print("[7/9] simulating active learning ...", flush=True)
    active = simulate_active_learning(
        bundle.train,
        pool,
        featurizer,
        ActiveLearningConfig(
            initial_labels=config.al_initial,
            batch_size=config.al_batch,
            rounds=config.al_rounds,
            n_bootstrap=config.al_bootstrap,
            seed=config.seed + 1,
        ),
    )
    _write_json(
        run_dir / "active_learning.json",
        {"summary": active.summary, "curve": active.curve},
    )

    # ------------------------------------------------------ 8. interpretability
    if verbose:
        print("[8/9] computing attention, Shapley values and ablations ...", flush=True)
    attention_payload: list[dict[str, Any]] = []
    shapley_payload: list[dict[str, Any]] = []
    for candidate in candidates[:3]:
        prediction = predict_materials(
            results[0].model,
            featurizer,
            results[0].scaler,
            [candidate.material],
            return_attention=True,
        )
        attention = prediction.get("attention")
        if attention:
            rollout = attention_rollout(attention, candidate.material)
            order = np.argsort(-rollout.importance[1:])[:10] + 1
            attention_payload.append(
                {
                    "material_id": candidate.material.id,
                    "label": candidate.material.label,
                    "tokens": rollout.tokens,
                    "importance": rollout.importance.tolist(),
                    "top_tokens": [
                        {"token": rollout.tokens[int(i)], "weight": float(rollout.importance[int(i)])}
                        for i in order
                    ],
                }
            )
        shapley = element_shapley(results, featurizer, candidate.material, "her")
        shapley_payload.append(shapley)
    permutation = permutation_importance(
        results, featurizer, bundle.test[:300], repeats=5, seed=config.seed + 2
    )
    interpret_payload = {
        "attention": attention_payload,
        "shapley": shapley_payload,
        "permutation": permutation,
    }
    _write_json(run_dir / "interpret.json", interpret_payload)

    # ------------------------------------------------------------- 9. figures
    if verbose:
        print("[9/9] writing figures and report ...", flush=True)
    if config.figures:
        from . import viz

        if results:
            _plot(
                run_dir,
                "training_curves.png",
                viz.plot_training_curves,
                [result.history.epochs for result in results],
            )
        test_prediction = ensemble_predict(results, featurizer, bundle.test)
        ood_prediction = ensemble_predict(results, featurizer, bundle.ood)
        test_truth = np.array([material.eg for material in bundle.test])
        ood_truth = np.array([material.eg for material in bundle.ood])
        _plot(
            run_dir,
            "parity_bandgap.png",
            viz.plot_parity,
            test_truth,
            test_prediction.eg,
            ood_truth,
            ood_prediction.eg,
        )
        _plot(
            run_dir,
            "parity_her.png",
            viz.plot_parity,
            np.array([material.her for material in bundle.test]),
            np.clip(test_prediction.her, 0.01, None),
            np.array([material.her for material in bundle.ood]),
            np.clip(ood_prediction.her, 0.01, None),
            title="HER parity",
            unit="mmol g-1 h-1",
            log_y=True,
        )
        _plot(
            run_dir,
            "screening.png",
            viz.plot_screening,
            pool_prediction.eg,
            pool_prediction.her,
            [(candidate.eg, candidate.her, candidate.material.formula) for candidate in candidates[:8]],
        )
        _plot(
            run_dir,
            "bias_audit.png",
            viz.plot_bias_audit,
            _counts(bundle.train),
            _counts(bundle.test),
            _counts(bundle.ood),
        )
        _plot(run_dir, "active_learning.png", viz.plot_active_learning, active.curve)
        _plot(
            run_dir,
            "uncertainty.png",
            viz.plot_uncertainty,
            test_prediction.eg_std,
            np.abs(test_truth - test_prediction.eg),
        )
        if attention_payload:
            first = attention_payload[0]
            _plot(
                run_dir,
                "attention.png",
                viz.plot_attention,
                first["tokens"],
                np.asarray(first["importance"]),
            )
        if shapley_payload:
            first_shapley = shapley_payload[0]
            _plot(
                run_dir,
                "shapley.png",
                viz.plot_shapley,
                first_shapley["attribution"],  # type: ignore[index]
                str(first_shapley["label"]),
            )

    metrics_payload = {
        "data": data_summary,
        "splits": {
            key: value
            for key, value in evaluations.items()
            if key not in ("baselines", "success_criteria")
        },
        "baselines": evaluations.get("baselines", {}),
        "success_criteria": success_criteria,
        "conformal": conformal,
        "ranking": funnel,
        "runtime_seconds": round(time.time() - started, 2),
    }
    _write_json(run_dir / "metrics.json", metrics_payload)
    generate_report(run_dir)

    store.close()
    if verbose:
        print(
            f"done in {time.time() - started:.1f}s -> {run_dir}/report.md\n"
            f"test Eg RMSE {evaluations['test']['bandgap']['rmse']:.3f} eV | "
            f"HER r {evaluations['test']['her']['pearson']:.3f}",
            flush=True,
        )
    return {
        "run_dir": str(run_dir),
        "test_bandgap_rmse": float(evaluations["test"]["bandgap"]["rmse"]),
        "test_her_pearson": float(evaluations["test"]["her"]["pearson"]),
        "ood_bandgap_rmse": float(evaluations["ood"]["bandgap"]["rmse"]),
        "ood_her_pearson": float(evaluations["ood"]["her"]["pearson"]),
        "candidates": len(candidates),
        "runtime_seconds": round(time.time() - started, 2),
    }


def load_run(run_dir: str | Path) -> tuple[dict[str, Any], Featurizer, list[TrainingResult]]:
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    featurizer = Featurizer.load(run_dir / "featurizer.json")
    ensemble_meta = json.loads((run_dir / "ensemble.json").read_text(encoding="utf-8"))
    results: list[TrainingResult] = []
    for item in ensemble_meta["models"]:
        model = GraphAugmentedTransformer.from_config(item["model_config"], seed=int(item["seed"]))
        model.load(str(run_dir / item["path"]))
        scaler = TargetScaler.from_dict(item["scaler"])
        results.append(
            TrainingResult(
                model=model,
                scaler=scaler,
                history=TrainingHistory(),
                config=TrainConfig(seed=int(item["seed"])),
            )
        )
    return config, featurizer, results
