"""Inspect an existing run: ensemble predictions and top candidates.

Run after `python -m photocat demo`:
    python examples/quickstart.py runs/demo
"""

from __future__ import annotations

import sys

from photocat.data.synth import load_materials
from photocat.models.ensemble import ensemble_predict
from photocat.pipeline import load_run
from photocat.ranking import RankConfig, screen_candidates


def main(run_dir: str = "runs/demo") -> None:
    _, featurizer, results = load_run(run_dir)
    materials = load_materials(f"{run_dir}/materials.csv")
    pool = [material for material in materials if material.split in ("test", "ood")]
    train = [material for material in materials if material.split == "train"]

    prediction = ensemble_predict(results, featurizer, pool)
    candidates, funnel = screen_candidates(
        pool, prediction.as_dict(), train, RankConfig(top_k=15)
    )
    print("screening funnel:", funnel)
    print(f"\n{'rank':>4}  {'material':30s}  {'Eg':>5}  {'HER':>6}  {'sim':>5}  {'score':>6}")
    for candidate in candidates:
        print(
            f"{candidate.rank:4d}  {candidate.material.label:30s}  "
            f"{candidate.eg:5.2f}  {candidate.her:6.2f}  "
            f"{candidate.similarity:5.2f}  {candidate.score:6.3f}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/demo")
