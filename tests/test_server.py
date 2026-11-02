"""HTTP-level tests: job lifecycle, progress, errors, key handling, export safety."""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_providers import KEYS, FakeProviders
import web.server as srv


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeProviders()
        cls.env = mock.patch.dict(os.environ, cls.fake.env())
        cls.env.start()
        for v in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
            os.environ.pop(v, None)
        cls.runs = Path(tempfile.mkdtemp(prefix="llmd_runs_"))
        cls.patch = mock.patch.object(srv, "RUNS", cls.runs)
        cls.patch.start()
        cls.httpd = srv.make_server(0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.fake.answers[("openai", "s")] = lambda p: "Paris"
        cls.fake.answers[("anthropic", "s")] = lambda p: "Paris."

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown(); cls.httpd.server_close(); cls.fake.stop(); cls.patch.stop(); cls.env.stop()

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def jcall(self, path, body=None):
        code, raw = self.call(path, body)
        return code, json.loads(raw)

    def run_job(self, payload, timeout=30):
        code, d = self.jcall("/api/run", payload)
        self.assertEqual(code, 202)
        seen = []
        end = time.time() + timeout
        while time.time() < end:
            _, j = self.jcall("/api/job/" + d["job_id"])
            seen.append(j["progress"])
            if j["status"] != "running":
                return j, seen
            time.sleep(0.05)
        self.fail("job did not finish")

    def test_page_providers_and_static_std(self):
        code, raw = self.call("/")
        self.assertEqual(code, 200); self.assertIn(b"LLM Disagreement Analysis", raw)
        _, p = self.jcall("/api/providers")
        self.assertTrue(all(isinstance(v, bool) for v in p["server_keys"].values()))
        code, svg = self.call("/api/std")
        self.assertEqual(svg.count(b"data-state="), 13)

    def test_inspect_csv(self):
        code, d = self.jcall("/api/inspect", {"csv": "a;b\n1;2\n3;4\n"})
        self.assertEqual((code, d["columns"], d["n_rows"]), (200, ["a", "b"], 2))
        code, d = self.jcall("/api/inspect", {"csv": ""})
        self.assertEqual(code, 400)

    def test_mock_job_completes_with_progress_and_is_labelled_simulated(self):
        j, seen = self.run_job({"data_mode": "sample", "models": ["mock:model-A", "mock:model-B"], "samples": 2})
        self.assertEqual(j["status"], "done")
        self.assertEqual(seen, sorted(seen))                               # progress never goes backwards
        r = j["result"]
        self.assertTrue(r["summary"]["simulated"])
        self.assertEqual(r["diagram"]["current"], "REPORT_DELIVERED")
        self.assertEqual(r["svg"].count("data-state="), 13)

    def test_custom_csv_job_with_real_provider_and_exports(self):
        body = {"data_mode": "csv", "csv": "id,question\nq1,Capital of France?\nq2,Capital of Peru?\n",
                "mapping": {"prompt_template": "Q: {question}", "id_column": "id", "task_type": "open"},
                "models": ["openai:s", "anthropic:s"], "keys": {"openai": KEYS["openai"], "anthropic": KEYS["anthropic"]},
                "samples": 1, "temperature": 0.2}
        j, _ = self.run_job(body)
        self.assertEqual(j["status"], "done", j.get("error"))
        res = j["result"]
        self.assertEqual(res["comparison"]["overall"]["n_compared"], 2)
        self.assertEqual(res["comparison"]["overall"]["disagree_n"], 0)
        self.assertFalse(res["summary"]["simulated"])
        # the sent prompts came from the template
        self.assertIn("Q: Capital of France?", {c["prompt"] for c in self.fake.calls})
        # keys are never echoed back
        blob = json.dumps(j)
        for k in KEYS.values():
            self.assertNotIn(k, blob)
        # downloads work and contain no keys
        for f in ["report.html", "responses.csv", "items.csv", "pairwise.csv", "summary.json", "report.md", "trace.json", "std.drawio"]:
            code, raw = self.call(f"/api/export/{res['run_id']}/{f}")
            self.assertEqual(code, 200, f)
            for k in KEYS.values():
                self.assertNotIn(k.encode(), raw)

    def test_example_csvs_are_served_and_whitelisted(self):
        for name in ["open_questions.csv", "sentiment_classification.csv", "prompts_only.csv"]:
            code, raw = self.call("/api/example/" + name)
            self.assertEqual(code, 200, name)
            self.assertGreater(raw.count(b"\n"), 8)
        for bad in ["/api/example/../../main.py", "/api/example/secrets.csv", "/api/example/..%2fmain.py"]:
            self.assertEqual(self.call(bad)[0], 404, bad)

    def test_typed_prompts_job(self):
        j, _ = self.run_job({"data_mode": "prompts", "prompts": "Capital of France?\nCapital of Peru?",
                             "models": ["openai:s", "anthropic:s"], "keys": {"openai": KEYS["openai"], "anthropic": KEYS["anthropic"]}})
        self.assertEqual(j["status"], "done")
        self.assertEqual(j["result"]["comparison"]["overall"]["n_items"], 2)

    def test_failure_modes_return_failed_jobs_with_a_diagram(self):
        cases = {
            "no key": ({"data_mode": "prompts", "prompts": "hi", "models": ["gemini:s"]}, "no_models"),
            "bad key": ({"data_mode": "prompts", "prompts": "hi", "models": ["openai:s"], "keys": {"openai": "sk-bad-key-0000000000"}}, "no_models"),
            "no models": ({"data_mode": "prompts", "prompts": "hi", "models": []}, "invalid_input"),
            "empty prompts": ({"data_mode": "prompts", "prompts": "  ", "models": ["mock:a"]}, "invalid_input"),
            "bad template": ({"data_mode": "csv", "csv": "a\n1\n", "mapping": {"prompt_template": "{zzz}"}, "models": ["mock:a"]}, "invalid_input"),
            "bad number": ({"data_mode": "sample", "models": ["mock:a"], "samples": "many"}, "invalid_input"),
        }
        for name, (payload, kind) in cases.items():
            j, _ = self.run_job(payload)
            self.assertEqual(j["status"], "failed", name)
            self.assertEqual(j["error"]["kind"], kind, name)
            self.assertEqual(j["svg"].count("data-state="), 13, name)    # failed runs still draw the full diagram
            self.assertNotIn("sk-bad-key-0000000000", json.dumps(j), name)

    def test_failed_run_trace_shows_where_it_stopped(self):
        j, _ = self.run_job({"data_mode": "prompts", "prompts": "hi", "models": ["openai:s"], "keys": {"openai": "sk-bad-key-0000000000"}})
        ev = [t["event"] for t in j["diagram"]["trace"]]
        self.assertEqual(ev[-2:], ["startAnalysis", "queryFailed"])
        self.assertEqual(j["diagram"]["current"], "DATASET_READY")

    def test_test_model_endpoint(self):
        _, d = self.jcall("/api/test-model", {"model": "openai:s", "key": KEYS["openai"]})
        self.assertTrue(d["ok"])
        _, d = self.jcall("/api/test-model", {"model": "openai:s", "key": "sk-bad-key-0000000000"})
        self.assertFalse(d["ok"]); self.assertNotIn("sk-bad-key-0000000000", d["message"])
        _, d = self.jcall("/api/test-model", {"model": "gemini:s"})
        self.assertFalse(d["ok"]); self.assertIn("API key", d["message"])
        _, d = self.jcall("/api/test-model", {"model": "mock:x"})
        self.assertTrue(d["simulated"])

    def test_export_path_safety(self):
        for path in ["/api/export/../../etc/passwd", "/api/export/20250101-000000-abcd/../../x", "/api/export/..%2f..%2fetc/passwd",
                     "/api/export/20250101-000000-abcd/secrets.env", "/api/export/20250101-000000-abcd/responses.csv",
                     "/api/job/../../x", "/nope"]:
            code, _ = self.call(path)
            self.assertEqual(code, 404, path)

    def test_bad_requests(self):
        req = urllib.request.Request(self.base + "/api/run", data=b"not json", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req)
        self.assertEqual(ctx.exception.code, 400)
        self.assertEqual(self.call("/api/job/unknown")[0], 404)


if __name__ == "__main__":
    unittest.main()
