import json
import tempfile
import unittest
from pathlib import Path

from photocat.pipeline import PipelineConfig, run_pipeline


class PipelineSmokeTest(unittest.TestCase):
    def test_fast_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            config = PipelineConfig(
                out_dir=directory,
                n_materials=320,
                epochs=1,
                ensemble_size=1,
                n_layers=1,
                d_model=32,
                n_heads=4,
                d_ff=48,
                gnn_dim=16,
                link_epochs=0,
                run_link_prediction=False,
                n_corpus=20,
                al_initial=20,
                al_batch=10,
                al_rounds=2,
                al_bootstrap=2,
                top_k=5,
                figures=False,
            )
            result = run_pipeline(config, verbose=False)
            run_dir = Path(directory)
            for name in ("metrics.json", "rankings.csv", "predictions.csv", "kg_metrics.json", "report.md"):
                self.assertTrue((run_dir / name).exists(), name)
            metrics = json.loads((run_dir / "metrics.json").read_text())
            self.assertIn("splits", metrics)
            self.assertIn("success_criteria", metrics)
            self.assertIn("test", metrics["splits"])
            self.assertTrue(Path(result["run_dir"]).exists())


if __name__ == "__main__":
    unittest.main()
