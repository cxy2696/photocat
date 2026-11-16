"""Command-line interface for photocat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .benchmarks import BenchmarkConfig, run_gap_benchmark, run_her_benchmark
from .data.synth import generate_dataset, load_materials, save_materials, summary
from .kg.corpus import generate_corpus, save_corpus
from .kg.store import build_kg
from .models.ensemble import ensemble_predict
from .pipeline import PipelineConfig, load_run, run_pipeline
from .ranking import RankConfig, screen_candidates
from .report import generate_report


def _demo(args: argparse.Namespace) -> int:
    if args.fast:
        n, epochs, ensemble, link_epochs = 3000, 12, 2, 5
        al_rounds, al_bootstrap = 8, 3
    else:
        n, epochs, ensemble, link_epochs = args.materials, args.epochs, args.ensemble, args.link_epochs
        al_rounds, al_bootstrap = 12, 5
    config = PipelineConfig(
        name="demo",
        out_dir=args.out,
        n_materials=n,
        seed=args.seed,
        epochs=epochs,
        ensemble_size=ensemble,
        top_k=args.top_k,
        link_epochs=link_epochs,
        run_link_prediction=not args.no_link,
        al_rounds=al_rounds,
        al_bootstrap=al_bootstrap,
        figures=not args.no_figures,
        mitigation=args.mitigation,
    )
    result = run_pipeline(config, verbose=True)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _data(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = generate_dataset(n=args.materials, seed=args.seed)
    save_materials(out / "materials.csv", bundle.materials)
    corpus = generate_corpus(bundle.materials, n=args.corpus, seed=args.seed)
    save_corpus(out / "corpus.jsonl", corpus)
    print(json.dumps(summary(bundle.materials), indent=2, ensure_ascii=False))
    print(f"wrote {out}/materials.csv, corpus.jsonl")
    return 0


def _kg(args: argparse.Namespace) -> int:
    data_dir = Path(args.data)
    materials = load_materials(data_dir / "materials.csv")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with build_kg(out, materials) as store:
        print(json.dumps(store.stats(), indent=2))
    return 0


def _screen(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    config, featurizer, results = load_run(run_dir)
    materials = load_materials(run_dir / "materials.csv")
    pool = [material for material in materials if material.split in ("test", "ood")]
    prediction = ensemble_predict(results, featurizer, pool)
    train = [material for material in materials if material.split == "train"]
    candidates, funnel = screen_candidates(
        pool, prediction.as_dict(), train, RankConfig(top_k=args.top_k)
    )
    path = run_dir / "screen.csv"
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        rows = [candidate.to_row() for candidate in candidates]
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(funnel, indent=2))
    for candidate in candidates[: args.top_k]:
        row = candidate.to_row()
        print(
            f"{row['rank']:3d} {row['label']:32s} {row['family']:16s} "
            f"Eg={row['pred_eg']:.2f} HER={row['pred_her']:.1f} "
            f"sim={row['novelty_similarity']:.2f} score={row['score']:.3f}"
        )
    print(f"\nwrote {path}")
    return 0


def _bench(args: argparse.Namespace) -> int:
    config = BenchmarkConfig(
        seed=args.seed,
        epochs=args.epochs,
        ensemble_size=args.ensemble,
        figures=not args.no_figures,
        verbose=True,
    )
    results = {}
    if args.dataset in ("gap", "all"):
        results["gap"] = run_gap_benchmark(
            BenchmarkConfig(**{**config.to_dict(), "out_dir": "runs/bench_gap"})
        )
    if args.dataset in ("her", "all"):
        results["her"] = run_her_benchmark(
            BenchmarkConfig(**{**config.to_dict(), "out_dir": "runs/bench_her"})
        )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


def _report(args: argparse.Namespace) -> int:
    path = generate_report(args.run)
    print(f"wrote {path}")
    return 0


def _info(_: argparse.Namespace) -> int:
    import numpy
    from importlib.metadata import PackageNotFoundError, version

    try:
        matplotlib_version = version("matplotlib")
    except PackageNotFoundError:
        matplotlib_version = "not installed (figures disabled)"
    payload = {
        "photocat": __version__,
        "numpy": numpy.__version__,
        "matplotlib": matplotlib_version,
        "python": __import__("sys").version.split()[0],
    }
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photocat",
        description="KG + graph-augmented transformer for photocatalytic materials discovery",
    )
    parser.add_argument("--version", action="version", version=f"photocat {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="run the full end-to-end pipeline")
    demo.add_argument("--materials", type=int, default=12000, help="number of library records")
    demo.add_argument("--epochs", type=int, default=30, help="training epochs per ensemble member")
    demo.add_argument("--ensemble", type=int, default=3, help="number of independently seeded models")
    demo.add_argument("--link-epochs", type=int, default=12, help="DistMult training epochs")
    demo.add_argument("--top-k", type=int, default=50, help="shortlist size")
    demo.add_argument("--seed", type=int, default=2025)
    demo.add_argument("--out", default="runs/demo")
    demo.add_argument("--mitigation", choices=("none", "balanced", "oversample"), default="none")
    demo.add_argument("--fast", action="store_true", help="small smoke run (~1-2 min)")
    demo.add_argument("--no-link", action="store_true", help="skip DistMult link prediction")
    demo.add_argument("--no-figures", action="store_true", help="skip matplotlib figures")
    demo.set_defaults(func=_demo)

    data = subparsers.add_parser("data", help="generate the synthetic library and corpus")
    data.add_argument("--materials", type=int, default=12000)
    data.add_argument("--corpus", type=int, default=600)
    data.add_argument("--seed", type=int, default=2025)
    data.add_argument("--out", default="data")
    data.set_defaults(func=_data)

    kg = subparsers.add_parser("kg", help="build the SQLite knowledge graph")
    kg.add_argument("--data", default="data")
    kg.add_argument("--out", default="data/kg.sqlite")
    kg.set_defaults(func=_kg)

    bench = subparsers.add_parser("bench", help="run the real-data benchmarks")
    bench.add_argument("--dataset", choices=("gap", "her", "all"), default="all")
    bench.add_argument("--epochs", type=int, default=24)
    bench.add_argument("--ensemble", type=int, default=2)
    bench.add_argument("--seed", type=int, default=2025)
    bench.add_argument("--no-figures", action="store_true")
    bench.set_defaults(func=_bench)

    screen = subparsers.add_parser("screen", help="re-rank candidates from an existing run")
    screen.add_argument("--run", default="runs/demo")
    screen.add_argument("--top-k", type=int, default=25)
    screen.set_defaults(func=_screen)

    report = subparsers.add_parser("report", help="regenerate report.md and figures")
    report.add_argument("--run", default="runs/demo")
    report.set_defaults(func=_report)

    info = subparsers.add_parser("info", help="show versions")
    info.set_defaults(func=_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
