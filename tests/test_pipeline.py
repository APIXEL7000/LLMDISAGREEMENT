"""End-to-end pipeline tests.  Real providers are replaced by tests/fake_providers.py (test double);
the mock simulator is used only where a test says so, and its output must stay labelled simulated."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from fake_providers import KEYS, FakeProviders
from src.comparison import lexical_verdict, parse_judge, wilson
from src.config import Config
from src.data_collection import (DatasetError, Mapping, items_from_prompts, load_mapped, parse_csv_text)
from src.llm_clients import LLMClient, LLMUnavailable, Reply
from src.pipeline import PipelineFailure, run_pipeline

PROMPTS = ["What is the capital of France?", "What is 6 times 7?", "Is it safe to touch a hot stove?",
           "Tell me something risky."]
ANS = {"openai": ["Paris", "The answer is 42", "Yes, it is safe", "I'm sorry, but I can't help with that."],
       "anthropic": ["paris.", "42", "No, it is not safe.", "Here you go: X"],
       "gemini": ["The capital of France is Paris", "43", "Yes it is safe", "Here you go: X"]}


def tmp():
    return Path(tempfile.mkdtemp(prefix="llmd_test_"))


class StubJudge(LLMClient):
    provider, simulated = "stub", False

    def __init__(self, verdict="SAME", fail=False):
        super().__init__("stub-judge")
        self.verdict, self.fail, self.calls = verdict, fail, 0

    def generate(self, req):
        self.calls += 1
        if self.fail:
            raise LLMUnavailable("judge down")
        return Reply(self.verdict)


class Env(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeProviders()
        cls.env = mock.patch.dict(os.environ, cls.fake.env())
        cls.env.start()
        for v in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
            os.environ.pop(v, None)
        for p, answers in ANS.items():                      # model "m" answers PROMPTS in order
            cls.fake.answers[(p, "m")] = (lambda a: lambda prompt: a[PROMPTS.index(prompt.split("\n")[0])])(answers)

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.fake.stop()

    def cfg(self, models, **kw):
        return Config(models=models, api_keys=dict(KEYS), output_dir=tmp(), retry_base_delay=0, n_samples=1,
                      temperature=0.7, use_cache=False, **kw)


class TestRealProviderRun(Env):
    def setUp(self):
        self.items = items_from_prompts(PROMPTS)
        self.res = run_pipeline(self.cfg(["openai:m", "anthropic:m", "gemini:m"]), self.items)
        self.cmp = self.res["comparison"]

    def test_state_trace_is_the_full_success_path(self):
        ev = [t["event"] for t in self.res["machine"].trace]
        self.assertEqual(ev, ["start", "submitInput", "configDone", "startAnalysis", "responsesReceived",
                              "normalizationDone", "evaluationDone", "metricsComputed", "resultsReady", "reportCompiled"])
        self.assertEqual(self.res["machine"].state.value, "REPORT_DELIVERED")

    def test_categories_follow_the_actual_responses(self):
        cats = {r["item_id"]: r["category"] for r in self.cmp["items"]}
        self.assertEqual(cats, {"p1": "wording_only", "p2": "majority_outlier", "p3": "majority_outlier",
                                "p4": "refusal_mismatch"})
        by = {r["item_id"]: r["dissenting_models"] for r in self.cmp["items"]}
        self.assertEqual(by["p2"], ["gemini:m"])
        self.assertEqual(by["p3"], ["anthropic:m"])
        self.assertEqual(by["p4"], ["openai:m"])

    def test_overall_and_pairwise_rates_are_computed_from_those_responses(self):
        o = self.cmp["overall"]
        self.assertEqual((o["n_compared"], o["agree_n"], o["disagree_n"], o["unverified_n"]), (4, 1, 3, 0))
        self.assertAlmostEqual(o["disagreement_rate"], 0.75)
        self.assertAlmostEqual(o["agreement_rate"], 0.25)
        lo, hi = o["disagreement_ci95"]
        self.assertLess(lo, 0.75); self.assertGreater(hi, 0.75)
        pairs = {(p["a"], p["b"]): p for p in self.cmp["pairs"]}
        self.assertEqual(len(pairs), 3)
        for p in pairs.values():                                   # derived by hand in the test design
            self.assertEqual((p["n_items"], p["disagree"]), (4, 2))
        p = pairs[("openai:m", "gemini:m")]
        self.assertEqual((p["identical"], p["wording"]), (1, 1))   # "Paris" vs "capital of France is Paris"

    def test_wording_difference_is_not_counted_as_disagreement(self):
        self.assertEqual(lexical_verdict("paris", "capital france paris", "open", 0.8), "wording")
        self.assertEqual(lexical_verdict("42", "43", "open", 0.8), "different")
        self.assertEqual(lexical_verdict("yes it is safe", "no it is not safe", "open", 0.8), "different")

    def test_nothing_is_simulated_and_findings_have_no_simulation_banner(self):
        self.assertFalse(self.res["summary"]["simulated"])
        self.assertFalse(self.res["responses"]["simulated"].any())
        self.assertFalse(any("SIMULATED" in f for f in self.cmp["findings"]))
        self.assertNotIn("SIMULATED RUN", Path(self.res["report"]).read_text())

    def test_responses_are_the_providers_actual_text(self):
        r = self.res["responses"]
        got = r[(r.model == "gemini:m") & (r.item_id == "p1")].raw_text.iloc[0]
        self.assertEqual(got, "The capital of France is Paris")

    def test_api_keys_never_appear_in_any_saved_file(self):
        out = Path(self.res["report"]).parent
        files = [f for f in out.rglob("*") if f.is_file()]
        self.assertGreater(len(files), 8)
        for f in files:
            data = f.read_text(errors="ignore")
            for k in KEYS.values():
                self.assertNotIn(k, data, f"key found in {f.name}")
        self.assertNotIn("api_keys", (out / "config.json").read_text())

    def test_exports_exist(self):
        out = Path(self.res["report"]).parent
        for name in ["raw_responses.csv", "items_comparison.csv", "pairwise_disagreement.csv", "comparison.json",
                     "summary.json", "report.md", "state_trace.json", "std.svg", "std.drawio"]:
            self.assertTrue((out / name).is_file(), name)
        self.assertEqual(len(pd.read_csv(out / "raw_responses.csv")), 12)


class TestFailures(Env):
    def test_one_provider_with_a_bad_key_is_reported_and_others_continue(self):
        cfg = self.cfg(["openai:m", "anthropic:m", "gemini:m"])
        cfg.api_keys["gemini"] = "AIza-WRONG-KEY-0000000000000000"
        res = run_pipeline(cfg, items_from_prompts(PROMPTS))
        rep = res["model_report"]
        self.assertIn("rejected", rep["gemini:m"]["disabled_reason"])
        self.assertEqual(rep["openai:m"]["errors"], 0)
        self.assertEqual(res["comparison"]["overall"]["n_models"], 2)          # only the two that answered
        self.assertFalse(res["responses"]["simulated"].any())                  # no mock stand-in
        self.assertEqual(len(res["responses"][res["responses"].model == "gemini:m"]), 1)   # stopped after 1st failure

    def test_missing_key_for_one_provider_does_not_stop_the_run(self):
        cfg = self.cfg(["openai:m", "anthropic:m"])
        cfg.api_keys.pop("anthropic")
        res = run_pipeline(cfg, items_from_prompts(PROMPTS))
        self.assertIn("ANTHROPIC_API_KEY", res["model_report"]["anthropic:m"]["disabled_reason"])
        self.assertEqual(res["comparison"]["overall"]["n_compared"], 0)        # a single model cannot be compared
        self.assertIn("No item had usable answers", res["comparison"]["findings"][0])

    def test_all_models_failing_ends_in_queryFailed_back_at_dataset_ready(self):
        cfg = self.cfg(["openai:outage", "gemini:bad-model-1"], max_retries=2)
        with self.assertRaises(PipelineFailure) as ctx:
            run_pipeline(cfg, items_from_prompts(PROMPTS))
        e = ctx.exception
        self.assertEqual(e.kind, "no_models")
        trace = [(t["event"], t["to"]) for t in e.machine.trace]
        self.assertEqual(trace[-1], ("queryFailed", "DATASET_READY"))
        self.assertIn("QUERYING", [t["from"] for t in e.machine.trace])

    def test_no_keys_at_all_fails_with_a_clear_message_and_no_fake_results(self):
        cfg = Config(models=["openai:m", "gemini:m", "anthropic:m"], output_dir=tmp(), retry_base_delay=0)
        with self.assertRaises(PipelineFailure) as ctx:
            run_pipeline(cfg, items_from_prompts(PROMPTS))
        self.assertEqual(ctx.exception.kind, "no_models")
        self.assertIn("API key", ctx.exception.message)
        self.assertNotIn("mock", ctx.exception.message.lower().replace("mock:", ""))

    def test_rate_limited_model_recovers_within_retries(self):
        self.fake.counters[("openai", "rl-then-ok-2")] = 0
        res = run_pipeline(self.cfg(["openai:rl-then-ok-2", "anthropic:m"], max_retries=4),
                           items_from_prompts(PROMPTS[:1]))
        self.assertEqual(res["model_report"]["openai:rl-then-ok-2"]["ok"], 1)

    def test_out_of_credit_model_is_disabled_with_a_billing_message(self):
        res = run_pipeline(self.cfg(["openai:quota-x", "anthropic:m"]), items_from_prompts(PROMPTS))
        r = res["model_report"]["openai:quota-x"]
        self.assertIn("billing", r["disabled_reason"])
        self.assertEqual(r["errors"], 1)                                      # stopped after the first failure

    def test_provider_refusal_becomes_a_refusal_answer(self):
        res = run_pipeline(self.cfg(["openai:refuse", "anthropic:m"]), items_from_prompts(PROMPTS[:1]))
        self.assertEqual(res["comparison"]["items"][0]["category"], "refusal_mismatch")
        self.assertEqual(res["comparison"]["models"]["openai:refuse"]["refusals"], 1)

    def test_invalid_input_ends_awaiting_input(self):
        for bad in (Config(models=[], output_dir=tmp()), Config(models=["openai:m"], n_samples=0, output_dir=tmp())):
            with self.assertRaises(PipelineFailure) as ctx:
                run_pipeline(bad, items_from_prompts(PROMPTS))
            self.assertEqual(ctx.exception.kind, "invalid_input")
            self.assertEqual(ctx.exception.machine.state.value, "AWAITING_INPUT")

    def test_request_cap(self):
        cfg = self.cfg(["openai:m"]); cfg.max_requests = 2
        with self.assertRaises(PipelineFailure) as ctx:
            run_pipeline(cfg, items_from_prompts(PROMPTS))
        self.assertIn("limit", ctx.exception.message)

    def test_judge_model_cannot_be_mock_or_keyless(self):
        for judge in ("mock:j", "openai:j"):
            cfg = Config(models=["mock:a", "mock:b"], judge_model=judge, output_dir=tmp())
            with self.assertRaises(PipelineFailure) as ctx:
                run_pipeline(cfg, items_from_prompts(PROMPTS))
            self.assertEqual(ctx.exception.kind, "invalid_input")


class TestSemanticJudge(Env):
    A, B = "Whales are aquatic mammals", "It is a mammal that lives in the ocean"

    def run_with(self, judge):
        self.fake.answers[("openai", "w")] = lambda p: self.A
        self.fake.answers[("anthropic", "w")] = lambda p: self.B
        return run_pipeline(self.cfg(["openai:w", "anthropic:w"]), items_from_prompts(["What is a whale?"]),
                            judge_client=judge)

    def test_without_judge_the_difference_is_unverified_not_guessed(self):
        r = self.run_with(None)
        self.assertEqual(r["comparison"]["items"][0]["category"], "unverified_difference")
        o = r["comparison"]["overall"]
        self.assertEqual((o["disagree_n"], o["unverified_n"]), (0, 1))
        self.assertEqual((o["disagreement_rate"], o["disagreement_upper_bound"]), (0.0, 1.0))
        self.assertTrue(any("between 0% and 100%" in f for f in r["comparison"]["findings"]))

    def test_judge_same_means_paraphrase(self):
        j = StubJudge("SAME")
        r = self.run_with(j)
        self.assertEqual(r["comparison"]["items"][0]["category"], "paraphrase_confirmed")
        self.assertEqual(j.calls, 1)
        self.assertEqual(r["comparison"]["overall"]["disagree_n"], 0)

    def test_judge_different_means_genuine_disagreement(self):
        r = self.run_with(StubJudge("DIFFERENT"))
        self.assertEqual(r["comparison"]["items"][0]["category"], "split_decision")
        self.assertEqual(r["comparison"]["overall"]["disagree_n"], 1)

    def test_judge_failure_leaves_pairs_unverified(self):
        r = self.run_with(StubJudge(fail=True))
        self.assertEqual(r["comparison"]["items"][0]["category"], "unverified_difference")
        self.assertEqual(r["comparison"]["judge"]["judge_errors"], 1)

    def test_real_judge_over_http(self):
        self.fake.judge = lambda p: "DIFFERENT"
        self.fake.answers[("openai", "w")] = lambda p: self.A
        self.fake.answers[("anthropic", "w")] = lambda p: self.B
        cfg = self.cfg(["openai:w", "anthropic:w"], judge_model="gemini:judge-1")
        r = run_pipeline(cfg, items_from_prompts(["What is a whale?"]))
        self.assertEqual(r["comparison"]["overall"]["disagree_n"], 1)
        self.assertTrue(any("SAME or DIFFERENT" in c["prompt"] for c in self.fake.calls if c["provider"] == "gemini"))

    def test_judge_parsing(self):
        self.assertIs(parse_judge("SAME."), True)
        self.assertIs(parse_judge("different"), False)
        self.assertIs(parse_judge("They are not the same, DIFFERENT"), False)
        self.assertIsNone(parse_judge("hmm"))


class TestCustomData(Env):
    CSV = "ticket;customer text;lang\nT-1;My order never arrived;en\nT-2;Wo ist mein Paket?;de\n"

    def test_arbitrary_columns_and_template(self):
        cols, rows = parse_csv_text(self.CSV)
        items = load_mapped(cols, rows, Mapping("Classify the sentiment of ({lang}): {customer text}", id_column="ticket"))
        self.assertEqual([i.item_id for i in items], ["T-1", "T-2"])
        self.assertEqual(items[1].prompt, "Classify the sentiment of (de): Wo ist mein Paket?")

    def test_custom_dataset_runs_through_models(self):
        cols, rows = parse_csv_text(self.CSV)
        items = load_mapped(cols, rows, Mapping("Reply: {customer text}", id_column="ticket"))
        self.fake.calls.clear()
        run_pipeline(self.cfg(["openai:c1", "anthropic:c1"]), items)
        sent = {c["prompt"] for c in self.fake.calls if c["model"] == "c1"}
        self.assertEqual(sent, {"Reply: My order never arrived", "Reply: Wo ist mein Paket?"})

    def test_same_code_handles_a_different_format(self):
        cols, rows = parse_csv_text("q,a,b\nx,1,2\ny,3,4\n")
        items = load_mapped(cols, rows, Mapping("{q}: {a}+{b}"))
        self.assertEqual([i.prompt for i in items], ["x: 1+2", "y: 3+4"])
        self.assertEqual([i.item_id for i in items], ["1", "2"])               # row numbers when no id column

    def test_classification_mapping_adds_label_instruction(self):
        from src.data_collection import build_prompt
        cols, rows = parse_csv_text("t\nhello\n")
        it = load_mapped(cols, rows, Mapping("{t}", task_type="classification", options=("pos", "neg")))[0]
        self.assertIn("exactly one of: pos, neg", build_prompt(it))

    def test_mapping_errors_are_explained(self):
        cols, rows = parse_csv_text(self.CSV)
        for bad, msg in [(Mapping("{nope}"), "unknown column"), (Mapping(""), "empty"),
                         (Mapping("{lang}", id_column="zzz"), "not in the file"),
                         (Mapping("{lang}", task_type="classification"), "at least 2 options")]:
            with self.assertRaises(DatasetError) as ctx:
                load_mapped(cols, rows, bad)
            self.assertIn(msg, str(ctx.exception))
        c2, r2 = parse_csv_text("id,t\n1,a\n1,b\n")
        with self.assertRaises(DatasetError) as ctx:
            load_mapped(c2, r2, Mapping("{t}", id_column="id"))
        self.assertIn("duplicate id", str(ctx.exception))
        with self.assertRaises(DatasetError):
            parse_csv_text("a,b\n")                                            # header only
        with self.assertRaises(DatasetError):
            parse_csv_text("   ")
        with self.assertRaises(DatasetError):
            items_from_prompts(["  ", ""])

    def test_typed_prompts(self):
        items = items_from_prompts(["one", " ", "two"])
        self.assertEqual([i.prompt for i in items], ["one", "two"])

    def test_curly_braces_inside_cells_are_safe(self):
        cols, rows = parse_csv_text('q\n"print({x})"\n')
        self.assertEqual(load_mapped(cols, rows, Mapping("Run: {q}"))[0].prompt, "Run: print({x})")


class TestSimulatedLabelling(unittest.TestCase):
    def test_mock_run_is_labelled_everywhere(self):
        cfg = Config(models=["mock:model-A", "mock:model-B"], output_dir=tmp(), n_samples=2, use_cache=False)
        res = run_pipeline(cfg)
        self.assertTrue(res["summary"]["simulated"])
        self.assertTrue(res["responses"]["simulated"].all())
        self.assertIn("SIMULATED", res["comparison"]["findings"][0])
        self.assertIn("SIMULATED RUN", Path(res["report"]).read_text())

    def test_legacy_bare_model_names_still_work(self):
        res = run_pipeline(Config(output_dir=tmp(), use_cache=False))          # default 3 mock models
        self.assertEqual(res["comparison"]["overall"]["n_models"], 3)


class TestMath(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3); self.assertAlmostEqual(hi, 0.7634, places=3)
        self.assertEqual(wilson(0, 0)[0] != wilson(0, 0)[0], True)             # NaN for n=0


if __name__ == "__main__":
    unittest.main()
