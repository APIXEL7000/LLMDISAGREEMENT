"""Plain-English explanations must be derived from the real numbers (no fabricated conclusions)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_providers import KEYS, FakeProviders
from src.config import Config
from src.data_collection import items_from_prompts
from src.explain import explain_results, level_for
from src.llm_clients import LLMUnavailable, LLMRequestError, list_ollama_models, make_client, GenRequest
from src.pipeline import run_pipeline


class TestExplain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeProviders()
        cls.env = mock.patch.dict(os.environ, cls.fake.env()); cls.env.start()
        for v in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_API_KEY"): os.environ.pop(v, None)
        cls.fake.answers[("openai", "e")] = lambda p: "42" if "6" in p else "Paris"
        cls.fake.answers[("anthropic", "e")] = lambda p: "The answer is 42 and that is all I will say about it today" if "6" in p else "paris."
        cls.fake.answers[("gemini", "e")] = lambda p: "43" if "6" in p else "I'm sorry, but I can't help with that."

    @classmethod
    def tearDownClass(cls): cls.env.stop(); cls.fake.stop()

    def run_it(self, models, prompts):
        cfg = Config(models=models, api_keys=dict(KEYS), output_dir=Path(tempfile.mkdtemp()), retry_base_delay=0, n_samples=1,
                     temperature=.5, use_cache=False)
        return run_pipeline(cfg, items_from_prompts(prompts))

    def test_explanation_numbers_match_the_results(self):
        r = self.run_it(["openai:e", "anthropic:e", "gemini:e"], ["What is 6 times 7?", "Capital of France?"])
        ex, o = r["comparison"]["explain"], r["comparison"]["overall"]
        self.assertIs(ex, r["comparison"]["explain"])
        by = {m["key"]: m for m in ex["metrics"]}
        self.assertEqual(by["agreement"]["value"], f"{100 * o['agreement_rate']:.0f}%")
        self.assertEqual(by["disagreement"]["value"], f"{100 * o['disagreement_rate']:.0f}%")
        self.assertIn(f"{o['n_compared']} of {o['n_items']}", ex["headline"])
        self.assertEqual(by["coverage"]["value"], f"{o['n_compared']} / {o['n_items']}")

    def test_examples_and_reasons_come_from_actual_items(self):
        r = self.run_it(["openai:e", "anthropic:e", "gemini:e"], ["What is 6 times 7?", "Capital of France?"])
        ex = r["comparison"]["explain"]
        dis = next(m for m in ex["metrics"] if m["key"] == "disagreement")["example"]
        self.assertIn("gemini:e said", dis)                    # real answers from this run
        self.assertIn("What is 6 times 7?", dis)
        reasons = {x["reason"] for x in ex["reasons"]}
        self.assertIn("Different safety rules", reasons)       # Q2: gemini refused
        self.assertIn("Real differences in knowledge or reasoning, or an ambiguous question", reasons)
        self.assertIn("Different answer length", reasons)      # anthropic's Q1 answer is >2x longer
        self.assertTrue(any("Most different pair" in n for n in ex["model_notes"]))   # 3 models -> 3 pairs -> a ranking is shown
        self.assertIn("not prove", ex["reasons_note"])

    def test_no_invented_reasons_when_everything_agrees(self):
        self.fake.answers[("openai", "same")] = lambda p: "Paris"
        self.fake.answers[("anthropic", "same")] = lambda p: "Paris"
        r = self.run_it(["openai:same", "anthropic:same"], ["Capital of France?", "Capital of France again?"])
        ex = r["comparison"]["explain"]
        self.assertEqual(ex["reasons"], [])
        self.assertEqual(ex["level"]["label"], "low")
        self.assertIn("agreed on 2 (100%)", ex["headline"])
        self.assertTrue(all(m["value"] != "n/a" for m in ex["metrics"] if m["key"] in ("agreement", "disagreement")))

    def test_small_sample_warning_and_levels(self):
        self.assertEqual([level_for(x) for x in (0.0, 0.09, 0.1, 0.29, 0.3, 1.0, None)], ["low", "low", "moderate", "moderate", "high", "high", None])
        r = self.run_it(["openai:e", "anthropic:e"], ["Capital of France?"])
        self.assertIn("first indication", r["comparison"]["explain"]["level"]["text"])

    def test_level_text_matches_its_level(self):
        for rate, must, must_not in ((0.05, "Low disagreement", "High disagreement"), (0.2, "Moderate disagreement", "Low disagreement:"),
                                     (0.6, "High disagreement", "Low disagreement:")):
            from src.explain import HOW_TO_READ
            lvl = level_for(rate)
            self.assertIn(must, HOW_TO_READ[lvl]); self.assertNotIn(must_not, HOW_TO_READ[lvl])
        r = self.run_it(["openai:e", "anthropic:e", "gemini:e"], ["What is 6 times 7?", "Capital of France?"])
        ex = r["comparison"]["explain"]
        self.assertIn(ex["level"]["label"].capitalize() + " disagreement:", ex["level"]["text"])

    def test_pair_card_value_is_a_short_percentage(self):
        r = self.run_it(["openai:e", "anthropic:e", "gemini:e"], ["What is 6 times 7?", "Capital of France?"])
        pair = next(m for m in r["comparison"]["explain"]["metrics"] if m["key"] == "pairs")
        self.assertTrue(pair["value"].endswith("%") and len(pair["value"]) <= 5)
        self.assertIn(" vs ", pair["meaning"])

    def test_nothing_comparable(self):
        r = self.run_it(["openai:e"], ["Capital of France?"])
        ex = r["comparison"]["explain"]
        self.assertIn("could not be measured", ex["headline"])
        self.assertEqual((ex["metrics"], ex["reasons"]), ([], []))

    def test_unverified_metric_states_both_bounds(self):
        self.fake.answers[("openai", "u")] = lambda p: "Whales are aquatic mammals"
        self.fake.answers[("anthropic", "u")] = lambda p: "It is a mammal that lives in the ocean"
        ex = self.run_it(["openai:u", "anthropic:u"], ["What is a whale?"])["comparison"]["explain"]
        u = next(m for m in ex["metrics"] if m["key"] == "unverified")
        self.assertIn("0%", u["reading"]); self.assertIn("100%", u["reading"])

    def test_simulated_runs_stay_flagged(self):
        r = run_pipeline(Config(models=["mock:a", "mock:b"], output_dir=Path(tempfile.mkdtemp()), n_samples=1, use_cache=False))
        self.assertTrue(r["comparison"]["explain"]["simulated"])

    def test_report_contains_plain_english_section(self):
        r = self.run_it(["openai:e", "anthropic:e", "gemini:e"], ["What is 6 times 7?", "Capital of France?"])
        html = (Path(r["report"]).parent / "report.html").read_text(encoding="utf-8")
        for needle in ["Summary in plain English", "What each number means", "Possible reasons for the disagreements", "Why disagreement matters"]:
            self.assertIn(needle, html)


class TestOllamaFriendlyErrors(unittest.TestCase):
    def test_not_running_gives_actionable_message(self):
        with mock.patch.dict(os.environ, {"OLLAMA_URL": "http://127.0.0.1:9"}):
            with self.assertRaises(LLMUnavailable) as c1:
                list_ollama_models(timeout=1)
            with self.assertRaises(LLMUnavailable) as c2:
                make_client("ollama:llama3", Config()).generate(GenRequest("i", "hi", 0, 0.5, 8))
        for exc in (c1.exception, c2.exception):
            self.assertIn("ollama serve", str(exc)); self.assertIn("http://127.0.0.1:9", str(exc))

    def test_model_not_installed_and_listing_against_a_fake_ollama(self):
        import json, threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def _send(self, code, obj):
                raw = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
            def do_GET(self): self._send(200, {"models": [{"name": "zeta:1b"}, {"name": "alpha:3b"}]})
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if body["model"] == "alpha:3b": self._send(200, {"response": "hello", "done": True})
                else: self._send(404, {"error": f"model '{body['model']}' not found"})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with mock.patch.dict(os.environ, {"OLLAMA_URL": f"http://127.0.0.1:{srv.server_address[1]}/api/generate"}):   # old URL form still works
                self.assertEqual(list_ollama_models(), ["alpha:3b", "zeta:1b"])
                req = GenRequest("i", "hi", 0, 0.5, 8)
                self.assertEqual(make_client("ollama:alpha:3b", Config()).generate(req).text, "hello")
                with self.assertRaises(LLMRequestError) as c:
                    make_client("ollama:missing", Config()).generate(req)
                self.assertIn("ollama pull missing", str(c.exception))
        finally:
            srv.shutdown(); srv.server_close()


if __name__ == "__main__":
    unittest.main()


class TestCheckSetupScript(unittest.TestCase):
    def test_runs_and_never_prints_keys(self):
        import subprocess
        secret = "super-secret-gemini-key-123456"
        r = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "check_setup.py")], capture_output=True, text=True,
                           env={**os.environ, "GEMINI_API_KEY": secret, "OLLAMA_URL": "http://127.0.0.1:9"})
        self.assertNotIn(secret, r.stdout + r.stderr)
        self.assertIn("Gemini key configured: yes", r.stdout)
        self.assertIn("ollama serve", r.stdout)                  # Ollama not running -> actionable fix
        self.assertEqual(r.returncode, 1)
