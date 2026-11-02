"""Real client code paths exercised against tests/fake_providers.py (a test double - see that file)."""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_providers import KEYS, FakeProviders
from src.config import Config
from src.llm_clients import (GenRequest, LLMAuthError, LLMQuotaError, LLMRateLimit, LLMRequestError, LLMUnavailable, MissingAPIKey,
                             build_clients, make_client, parse_spec)
from src.sampling import _generate_with_retry
from src.secrets import get_key, key_status, load_dotenv, redact

REQ = GenRequest("i1", "What is 2+2?", 0, 0.7, 64)
PROVS = ["openai", "anthropic", "gemini", "ollama-cloud"]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeProviders()
        cls.env = mock.patch.dict(os.environ, cls.fake.env())
        cls.env.start()
        for v in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
            os.environ.pop(v, None)

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()
        cls.fake.stop()

    def cfg(self, **kw):
        return Config(api_keys=dict(KEYS), retry_base_delay=0, **kw)


class TestProviders(Base):
    def test_success_for_each_provider(self):
        for p in PROVS:
            self.fake.answers[(p, "m1")] = lambda prompt: "four"
            reply = make_client(f"{p}:m1", self.cfg()).generate(REQ)
            self.assertEqual((reply.text, reply.blocked), ("four", False), p)

    def test_request_shapes(self):
        for p in PROVS:
            make_client(f"{p}:shape", self.cfg()).generate(REQ)
        by = {c["provider"]: c["body"] for c in self.fake.calls if c["model"] == "shape"}
        self.assertEqual(by["openai"]["messages"][0]["content"], "What is 2+2?")
        self.assertIn("max_completion_tokens", by["openai"])
        self.assertEqual(by["anthropic"]["max_tokens"], 64)
        self.assertEqual(by["gemini"]["generationConfig"]["maxOutputTokens"], 64)
        self.assertEqual(by["ollama-cloud"]["prompt"], "What is 2+2?")
        self.assertEqual(by["ollama-cloud"]["options"]["num_predict"], 64)

    def test_wrong_key_is_an_auth_error_and_never_leaks_the_key(self):
        for p in PROVS:
            bad = "sk-WRONG-key-abcdefghijklmnop"
            with self.assertRaises(LLMAuthError) as ctx:
                make_client(f"{p}:m1", Config(api_keys={p: bad})).generate(REQ)
            self.assertNotIn(bad, str(ctx.exception))

    def test_bad_model_is_a_request_error_not_retryable(self):
        for p in PROVS:
            with self.assertRaises(LLMRequestError):
                make_client(f"{p}:bad-model-x", self.cfg()).generate(REQ)

    def test_outage_is_retryable_unavailable(self):
        for p in PROVS:
            with self.assertRaises(LLMUnavailable) as ctx:
                make_client(f"{p}:outage", self.cfg()).generate(REQ)
            self.assertTrue(ctx.exception.retryable)

    def test_rate_limit_is_retried_then_succeeds(self):
        for p in PROVS:
            self.fake.counters[(p, "rl-then-ok-1")] = 0
            c = make_client(f"{p}:rl-then-ok-1", self.cfg())
            with self.assertRaises(LLMRateLimit):                          # a single attempt fails ...
                c.generate(REQ)
            self.fake.counters[(p, "rl-then-ok-1")] = 0
            reply = _generate_with_retry(c, REQ, self.cfg(max_retries=3))  # ... the retry loop recovers
            self.assertEqual(reply.text, "OK")

    def test_out_of_credit_is_a_billing_error_and_is_not_retried(self):
        self.fake.counters[("openai", "quota-out")] = 0
        c = make_client("openai:quota-out", self.cfg())
        with self.assertRaises(LLMQuotaError) as ctx:
            _generate_with_retry(c, REQ, self.cfg(max_retries=4))
        self.assertIn("billing", str(ctx.exception))
        self.assertEqual(self.fake.counters[("openai", "quota-out")], 1)      # exactly one attempt, no pointless retries

    def test_rate_limit_gives_up_after_max_retries(self):
        c = make_client("openai:rl-forever", self.cfg())
        with self.assertRaises(LLMRateLimit):
            _generate_with_retry(c, REQ, self.cfg(max_retries=3))

    def test_refusals_and_blocks_are_flagged_not_errors(self):
        for p in ("openai", "anthropic", "gemini"):          # Ollama has no block signal; its refusals are caught as text
            reply = make_client(f"{p}:refuse", self.cfg()).generate(REQ)
            self.assertTrue(reply.blocked, p)

    def test_openai_retries_without_temperature_when_model_rejects_it(self):
        self.fake.calls.clear()
        reply = make_client("openai:no-temp", self.cfg()).generate(REQ)
        self.assertEqual(reply.text, "OK")
        sent = [c["body"] for c in self.fake.calls if c["model"] == "no-temp"]
        self.assertEqual(len(sent), 2)
        self.assertIn("temperature", sent[0])
        self.assertNotIn("temperature", sent[1])


class TestKeys(Base):
    def test_missing_key_raises_and_never_falls_back_to_mock(self):
        with self.assertRaises(MissingAPIKey):
            make_client("openai:m1", Config())
        clients, unavailable = build_clients(Config(models=["openai:m1", "gemini:m1"]), [])
        self.assertEqual(clients, {})
        self.assertEqual(set(unavailable), {"openai:m1", "gemini:m1"})
        self.assertIn("OPENAI_API_KEY", unavailable["openai:m1"])

    def test_key_sources_and_status_are_booleans(self):
        with mock.patch.dict(os.environ, {"GOOGLE_API_KEY": "g-env-key-123"}):
            self.assertEqual(get_key("gemini"), "g-env-key-123")
            self.assertEqual(get_key("gemini", " typed-key-1 "), "typed-key-1")      # typed key wins
            self.assertTrue(all(isinstance(v, bool) for v in key_status().values()))
            self.assertNotIn("g-env-key-123", str(key_status()))

    def test_dotenv_loading_does_not_override_environment(self):
        import tempfile
        f = Path(tempfile.mkdtemp()) / ".env"
        f.write_text('# c\nOPENAI_API_KEY="from-file-123456"\nFOO_TEST=bar\n')
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "already-set-123"}, clear=False):
            os.environ.pop("FOO_TEST", None)
            load_dotenv(f)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "already-set-123")
            self.assertEqual(os.environ["FOO_TEST"], "bar")
            os.environ.pop("FOO_TEST", None)

    def test_redact(self):
        self.assertEqual(redact("key=abcdef123456 failed", "abcdef123456"), "key=*** failed")
        self.assertNotIn("sk-abcdefghijklmn", redact("x sk-abcdefghijklmn y"))

    def test_spec_parsing(self):
        self.assertEqual(parse_spec("openai:gpt-x"), ("openai", "gpt-x"))
        self.assertEqual(parse_spec("model-A", "mock"), ("mock", "model-A"))
        self.assertEqual(parse_spec("ollama:llama3:8b"), ("ollama", "llama3:8b"))
        self.assertEqual(parse_spec("ollama-cloud:gpt-oss:120b"), ("ollama-cloud", "gpt-oss:120b"))

    def test_ollama_cloud_needs_a_key_but_local_ollama_does_not(self):
        with self.assertRaises(MissingAPIKey) as ctx:
            make_client("ollama-cloud:m", Config())
        self.assertIn("OLLAMA_API_KEY", str(ctx.exception))
        make_client("ollama:llama3", Config())                       # local: builds without any key


if __name__ == "__main__":
    unittest.main()
