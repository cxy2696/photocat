import tempfile
import unittest
from pathlib import Path

from photocat.data.synth import generate_dataset
from photocat.kg.corpus import generate_corpus
from photocat.kg.extraction import evaluate_extraction, extract, load_extractions
from photocat.kg.link_prediction import train_link_predictor
from photocat.kg.reasoning import explain, run_reasoning
from photocat.kg.store import build_kg
from photocat.ranking import RankConfig


class KnowledgeGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = generate_dataset(n=250, seed=5)
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tempdir.name) / "kg.sqlite"
        cls.store = build_kg(cls.path, cls.bundle.materials, cls.bundle.anchors)
        cls.corpus = generate_corpus(cls.bundle.materials, cls.bundle.anchors, n=30, seed=5)

    @classmethod
    def tearDownClass(cls):
        cls.store.close()
        cls.tempdir.cleanup()

    def test_store_and_query(self):
        stats = self.store.stats()
        self.assertGreater(stats["triples"], 1000)
        rows = self.store.query_materials(eg=(1.23, 3.0), her_min=6.0, limit=5)
        for row in rows:
            self.assertGreaterEqual(row["eg"], 1.23)
            self.assertLessEqual(row["eg"], 3.0)
            self.assertGreaterEqual(row["her"], 6.0)
        paths = self.store.find_paths("material:pc-000001", "application:hydrogen_evolution", 2)
        self.assertTrue(paths)

    def test_extraction(self):
        predictions = [extract(record.text) for record in self.corpus]
        metrics = evaluate_extraction(self.corpus, predictions)
        self.assertGreater(metrics["relations"]["f1"], 0.9)
        self.assertGreater(metrics["entities"]["f1"], 0.9)
        labels = {
            material.label: material
            for material in list(self.bundle.materials) + list(self.bundle.anchors)
        }
        linking = load_extractions(self.store, self.corpus, labels)
        self.assertGreater(linking["link_rate"], 0.9)

    def test_reasoning_and_explanation(self):
        result = run_reasoning(self.store, RankConfig(), max_heterojunctions=5)
        self.assertGreater(result.stats["candidates"], 0)
        self.assertTrue(result.heterojunctions)
        explanation = explain(self.store, result.candidates[0]["material_id"])
        self.assertIn("summary", explanation)
        self.assertTrue(explanation["direct_triples"])

    def test_link_prediction(self):
        result = train_link_predictor(self.store, epochs=2, dim=8, full_vocab_eval=5)
        self.assertIn("mrr", result.metrics)
        self.assertGreaterEqual(result.metrics["mrr"], 0.0)
        self.assertLessEqual(result.metrics["mrr"], 1.0)


if __name__ == "__main__":
    unittest.main()
