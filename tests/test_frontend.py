"""Runs the page's real JavaScript (Node.js, stub DOM) against real job results: success, mock run and
failed run.  Checks rendering, the simulated banner, error advice, key hiding and HTML escaping.
Skipped automatically when Node.js is not installed."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_providers import KEYS, FakeProviders
import web.server as srv

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js not installed")
class TestFrontend(unittest.TestCase):
    def test_page_renders_real_results(self):
        fake = FakeProviders()
        saved = dict(os.environ)
        try:
            os.environ.update(fake.env())
            for v in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
                os.environ.pop(v, None)
            srv.RUNS = Path(tempfile.mkdtemp())
            fake.answers[("openai", "s")] = lambda p: "42"
            fake.answers[("anthropic", "s")] = lambda p: "The answer is 42"
            fake.answers[("gemini", "s")] = lambda p: "43"

            def run(payload):
                jid = srv.start_job(payload)
                while srv.JOBS[jid]["status"] == "running":
                    time.sleep(0.05)
                return srv.JOBS[jid]

            jobs = {"ok": run({"data_mode": "prompts", "prompts": "What is 6 times 7?\nWhat is 40 plus 2?",
                               "models": ["openai:s", "anthropic:s", "gemini:s"], "keys": KEYS, "samples": 1}),
                    "bad": run({"data_mode": "prompts", "prompts": "hi", "models": ["openai:s"],
                                "keys": {"openai": "sk-bad-key-0000000000"}}),
                    "sample": run({"data_mode": "sample", "models": ["mock:model-A", "mock:model-B", "mock:model-C"],
                                   "samples": 3})}
        finally:
            os.environ.clear(); os.environ.update(saved); fake.stop()
        tmp = Path(tempfile.mkdtemp())
        (tmp / "jobs.json").write_text(json.dumps(jobs, default=str))
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        (tmp / "page.js").write_text(re.search(r"<script>(.*)</script>", html, re.S).group(1))
        r = subprocess.run(["node", str(ROOT / "tests" / "frontend_smoke.js")], capture_output=True, text=True,
                           env={**os.environ, "JOBS_JSON": str(tmp / "jobs.json"), "PAGE_JS": str(tmp / "page.js")})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("smoke checks passed", r.stdout)


if __name__ == "__main__":
    unittest.main()
