"""Unit tests (standard library unittest):  python -m unittest discover -s tests -v"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import Config, ConfigError
from src.data_collection import DatasetError, Item, build_prompt, load_dataset, parse_votes
from src.disagreement import (analyze_item, cluster_map, js_divergence, normalized_entropy,
                              shannon_entropy, total_variation)
from src.normalization import INVALID, normalize_choice, normalize_open


class TestMath(unittest.TestCase):
    def test_entropy(self):
        self.assertAlmostEqual(shannon_entropy([5, 5]), 1.0)
        self.assertEqual(shannon_entropy([10]), 0.0)
        self.assertAlmostEqual(normalized_entropy([1, 1, 1], 3), 1.0)

    def test_js(self):
        self.assertAlmostEqual(js_divergence([.5, .5], [.5, .5]), 0.0)
        self.assertAlmostEqual(js_divergence([1, 0], [0, 1]), 1.0)
        self.assertAlmostEqual(total_variation([1, 0], [0, 1]), 1.0)

    def test_clusters(self):
        m = cluster_map(["william shakespeare", "shakespeare", "marlowe"], "open", 0.5)
        self.assertEqual(m["william shakespeare"], m["shakespeare"])
        self.assertNotEqual(m["marlowe"], m["shakespeare"])


class TestNormalization(unittest.TestCase):
    OPTS = ("entailment", "neutral", "contradiction")

    def test_choice(self):
        self.assertEqual(normalize_choice("The answer is Neutral.", self.OPTS), "neutral")
        self.assertEqual(normalize_choice("ENTAILMENT", self.OPTS), "entailment")
        self.assertEqual(normalize_choice("no idea", self.OPTS), INVALID)
        self.assertEqual(normalize_choice("not sarcastic", ("sarcastic", "not_sarcastic")),
                         "not_sarcastic")

    def test_open(self):
        self.assertEqual(normalize_open("The answer is Paris."), "paris")
        self.assertEqual(normalize_open("  the Pacific   Ocean! "), "pacific ocean")


class TestDataCollection(unittest.TestCase):
    def test_parse_votes(self):
        self.assertEqual(parse_votes("a:7|b:3"), {"a": 7, "b": 3})
        with self.assertRaises(ValueError):
            parse_votes("a7")

    def _write(self, text):
        d = tempfile.mkdtemp()
        p = Path(d) / "d.csv"
        p.write_text(text, encoding="utf-8")
        return p

    def test_invalid_dataset(self):
        bad = self._write("id,task_type,instruction,text,options,human_votes,reference\n"
                          "x,classification,i,t,a|b,c:3,\nx,open,i,t,,,\n")
        with self.assertRaises(DatasetError) as ctx:
            load_dataset(bad)
        self.assertIn("duplicate id", str(ctx.exception))
        self.assertIn("not in options", str(ctx.exception))
        with self.assertRaises(DatasetError):
            load_dataset(Path("does_not_exist.csv"))

    def test_sample_dataset_loads(self):
        items = load_dataset(Config().data_path)
        self.assertEqual(len(items), 26)
        self.assertIn("Answer with exactly one of", build_prompt(items[0]))


class TestDetection(unittest.TestCase):
    def _group(self, answers):
        return pd.DataFrame({"item_id": "q", "model": [f"m{i % 2}" for i in range(len(answers))],
                             "answer": answers, "valid": True})

    def test_flagging_and_overconfidence(self):
        cfg = Config()
        item = Item("q", "classification", "i", "t", ("a", "b"), {"a": 5, "b": 5})
        r = analyze_item(item, self._group(["a"] * 10), cfg)       # LLM certain, humans split
        self.assertEqual(r["agreement"], 1.0)
        self.assertFalse(r["disagreement"])
        self.assertTrue(r["overconfident"])
        r = analyze_item(item, self._group(["a"] * 5 + ["b"] * 5), cfg)
        self.assertTrue(r["disagreement"])
        self.assertAlmostEqual(r["js_divergence"], 0.0)

    def test_config_validation(self):
        Config(n_samples=1, temperature=0).validate()          # one response per model is valid (cross-model comparison)
        for bad in (Config(n_samples=0), Config(n_samples=3, temperature=0),   # repeated samples need temperature > 0
                    Config(temperature=3), Config(models=[]), Config(backend="nope")):
            with self.assertRaises(ConfigError):
                bad.validate()


if __name__ == "__main__":
    unittest.main()
