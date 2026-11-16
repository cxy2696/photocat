"""Train a minimal model and make predictions (about 20 seconds on CPU)."""

from __future__ import annotations

import numpy as np

from photocat.data.dataset import Featurizer
from photocat.data.synth import generate_dataset
from photocat.metrics import pearson, rmse
from photocat.training import TrainConfig, predict_materials, train_ensemble


def main() -> None:
    bundle = generate_dataset(n=1200, seed=42)
    featurizer = Featurizer.fit(bundle.train)
    results = train_ensemble(
        [TrainConfig(epochs=8, batch_size=64, verbose=False, seed=1)],
        featurizer,
        bundle.train,
        bundle.val,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=48,
        gnn_dim=16,
        verbose=False,
    )
    prediction = predict_materials(results[0].model, featurizer, results[0].scaler, bundle.test)
    truth = np.array([[m.eg, m.her] for m in bundle.test])
    print(f"test bandgap RMSE: {rmse(truth[:, 0], prediction['eg']):.3f} eV")
    print(f"test HER Pearson: {pearson(truth[:, 1], prediction['her']):.3f}")


if __name__ == "__main__":
    main()
