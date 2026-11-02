"""TEST DOUBLE ONLY.  A local HTTP server that imitates the request/response shapes of the OpenAI,
Anthropic and Gemini APIs so the real client code paths (auth, errors, retries, parsing) can be
tested without credentials.  It proves the clients behave correctly against *these* shapes; it is
NOT evidence about the real services.  Behaviour is chosen by the model name:
  quota*      -> 429 insufficient_quota (out of credit; must NOT be retried)
  bad-model*  -> 400     outage* -> 500      rl-forever* -> 429      rl-then-ok* -> 429 twice then OK
  refuse*     -> provider refusal / block      no-temp* (openai) -> 400 when 'temperature' is sent
  anything else -> answers[(provider, model)](prompt)
"""
import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEYS = {"openai": "sk-test-openai-key-123456", "anthropic": "sk-ant-test-key-123456",
        "gemini": "AIzaTestGeminiKey1234567890abcdef", "ollama-cloud": "ollama-cloud-test-key-123456"}


class FakeProviders:
    def __init__(self):
        self.answers, self.calls, self.counters = {}, [], Counter()
        self.judge = lambda prompt: "SAME"
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake._handle(self, body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def env(self):
        return {"OPENAI_BASE_URL": self.base + "/openai", "ANTHROPIC_BASE_URL": self.base + "/anthropic",
                "GEMINI_BASE_URL": self.base + "/gemini",
                "OLLAMA_CLOUD_URL": self.base + "/ollama-cloud"}

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    # ---- helpers ----
    def _reply(self, h, code, obj, headers=None):
        raw = json.dumps(obj).encode()
        h.send_response(code)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(raw)))
        for k, v in (headers or {}).items():
            h.send_header(k, v)
        h.end_headers()
        h.wfile.write(raw)

    def _handle(self, h, body):
        path = h.path
        if path.startswith("/openai"):
            prov, model = "openai", body.get("model", "")
            auth_ok = h.headers.get("Authorization") == f"Bearer {KEYS['openai']}"
            prompt = body["messages"][0]["content"]
        elif path.startswith("/ollama-cloud"):
            prov, model = "ollama-cloud", body.get("model", "")
            auth_ok = h.headers.get("Authorization") == f"Bearer {KEYS['ollama-cloud']}"
            prompt = body["prompt"]
        elif path.startswith("/anthropic"):
            prov, model = "anthropic", body.get("model", "")
            auth_ok = h.headers.get("x-api-key") == KEYS["anthropic"] and bool(h.headers.get("anthropic-version"))
            prompt = body["messages"][0]["content"]
        else:
            prov, model = "gemini", path.split("/models/")[1].split(":")[0]
            auth_ok = h.headers.get("x-goog-api-key") == KEYS["gemini"]
            prompt = body["contents"][0]["parts"][0]["text"]
        self.calls.append({"provider": prov, "model": model, "prompt": prompt, "body": body})
        if not auth_ok:
            return self._reply(h, 401, {"error": {"message": "Incorrect API key provided"}})
        self.counters[(prov, model)] += 1
        n = self.counters[(prov, model)]
        if model.startswith("bad-model"):
            return self._reply(h, 400, {"error": {"message": f"model '{model}' not found"}})
        if model.startswith("quota"):
            return self._reply(h, 429, {"error": {"message": "You exceeded your current quota, please check your plan and billing details.",
                                                  "type": "insufficient_quota", "code": "insufficient_quota"}})
        if model.startswith("outage"):
            return self._reply(h, 500, {"error": {"message": "internal error"}})
        if model.startswith("rl-forever") or (model.startswith("rl-then-ok") and n <= 2):
            return self._reply(h, 429, {"error": {"message": "rate limit exceeded"}}, {"Retry-After": "0"})
        if prov == "openai" and model.startswith("no-temp") and "temperature" in body:
            return self._reply(h, 400, {"error": {"message": "Unsupported value: 'temperature' does not support 0.7"}})
        if model.startswith("refuse"):
            if prov == "openai":
                return self._reply(h, 200, {"choices": [{"message": {"content": None, "refusal": "I can't help with that."},
                                                        "finish_reason": "stop"}]})
            if prov == "anthropic":
                return self._reply(h, 200, {"content": [], "stop_reason": "refusal"})
            return self._reply(h, 200, {"promptFeedback": {"blockReason": "SAFETY"}})
        if "SAME or DIFFERENT" in prompt:
            text = self.judge(prompt)
        else:
            text = self.answers.get((prov, model), lambda p: "OK")(prompt)
        if prov == "ollama-cloud":
            return self._reply(h, 200, {"model": model, "response": text, "done": True})
        if prov == "openai":
            return self._reply(h, 200, {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})
        if prov == "anthropic":
            return self._reply(h, 200, {"content": [{"type": "text", "text": text}], "stop_reason": "end_turn"})
        return self._reply(h, 200, {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]})
