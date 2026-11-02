"""Downloadable HTML report + the example CSVs shipped for testing."""
import os
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_providers import KEYS, FakeProviders
from src.config import Config
from src.data_collection import Mapping, build_prompt, load_mapped, parse_csv_text
from src.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "data" / "examples"


class Tags(HTMLParser):
    def __init__(self):
        super().__init__(); self.stack, self.errors, self.text = [], [], []

    VOID = {"meta", "br", "hr", "img", "input", "link", "path", "rect", "circle", "marker", "stop"}

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID: self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        if tag in self.VOID: return
        if not self.stack or self.stack[-1] != tag: self.errors.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")
        else: self.stack.pop()

    def handle_data(self, d): self.text.append(d)


def tmp(): return Path(tempfile.mkdtemp(prefix="llmd_rep_"))


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeProviders()
        cls.env = mock.patch.dict(os.environ, cls.fake.env()); cls.env.start()
        for v in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"): os.environ.pop(v, None)
        cls.fake.answers[("openai", "r")] = lambda p: "42 <script>alert(1)</script>"
        cls.fake.answers[("anthropic", "r")] = lambda p: "42 <script>alert(1)</script>"
        cls.fake.answers[("gemini", "r")] = lambda p: "43"

    @classmethod
    def tearDownClass(cls): cls.env.stop(); cls.fake.stop()

    def run_real(self):
        from src.data_collection import items_from_prompts
        cfg = Config(models=["openai:r", "anthropic:r", "gemini:r"], api_keys=dict(KEYS), output_dir=tmp(), retry_base_delay=0,
                     n_samples=1, temperature=0.5, use_cache=False)
        return run_pipeline(cfg, items_from_prompts(["What is 6*7? <b>x</b>", "What is 40+2?"]))

    def test_report_is_written_complete_and_well_formed(self):
        res = self.run_real()
        f = Path(res["report"]).parent / "report.html"
        self.assertTrue(f.is_file())
        html = f.read_text(encoding="utf-8")
        for needle in ["Key findings", "Pairwise disagreement", "Categories", "How disagreement was measured", "Limitations",
                       "Pipeline state trace", "openai:r", "gemini:r", "majority outlier", "no API keys"]:
            self.assertIn(needle, html, needle)
        self.assertEqual(html.count('data-state="'), 13)                      # the full STD is embedded
        t = Tags(); t.feed(html)
        self.assertEqual((t.errors, t.stack), ([], []), "report HTML is not well-formed")

    def test_report_numbers_come_from_the_results(self):
        res = self.run_real()
        html = (Path(res["report"]).parent / "report.html").read_text(encoding="utf-8")
        o = res["comparison"]["overall"]
        self.assertEqual((o["n_compared"], o["disagree_n"]), (2, 2))
        self.assertIn("100%", html)                                            # 2 of 2 confirmed disagreement
        self.assertNotIn("SIMULATED RUN", html)

    def test_report_escapes_hostile_content_and_hides_keys(self):
        res = self.run_real()
        html = (Path(res["report"]).parent / "report.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertNotIn("<b>x</b>", html)
        for k in KEYS.values(): self.assertNotIn(k, html)

    def test_simulated_banner_in_report(self):
        res = run_pipeline(Config(models=["mock:a", "mock:b"], output_dir=tmp(), use_cache=False, n_samples=1))
        html = (Path(res["report"]).parent / "report.html").read_text(encoding="utf-8")
        self.assertIn("SIMULATED RUN", html)


class TestExamples(unittest.TestCase):
    MAPS = {
        "open_questions.csv": Mapping("{question}", id_column="id", reference_column="reference"),
        "sentiment_classification.csv": Mapping("Classify the sentiment of this product review: {review}", id_column="id",
                                                task_type="classification", options=("positive", "negative", "neutral")),
        "prompts_only.csv": Mapping("{prompt}"),
    }

    def test_every_example_loads_with_its_documented_mapping(self):
        counts = {"open_questions.csv": 20, "sentiment_classification.csv": 15, "prompts_only.csv": 10}
        for name, mp in self.MAPS.items():
            cols, rows = parse_csv_text((EX / name).read_text(encoding="utf-8"))
            items = load_mapped(cols, rows, mp)
            self.assertEqual(len(items), counts[name], name)
            self.assertTrue(all(i.prompt for i in items))

    def test_classification_prompt_lists_the_labels(self):
        cols, rows = parse_csv_text((EX / "sentiment_classification.csv").read_text(encoding="utf-8"))
        p = build_prompt(load_mapped(cols, rows, self.MAPS["sentiment_classification.csv"])[3])
        self.assertIn("exactly one of: positive, negative, neutral", p)
        self.assertIn("Oh great, another update", p)

    def test_examples_run_end_to_end_with_labelled_mock(self):
        for name, mp in self.MAPS.items():
            cols, rows = parse_csv_text((EX / name).read_text(encoding="utf-8"))
            items = load_mapped(cols, rows, mp)
            res = run_pipeline(Config(models=["mock:model-A", "mock:model-B"], output_dir=tmp(), n_samples=1, use_cache=False), items)
            self.assertEqual(res["comparison"]["overall"]["n_items"], len(items), name)
            self.assertTrue(res["summary"]["simulated"], name)


if __name__ == "__main__":
    unittest.main()
