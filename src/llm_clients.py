"""LLM back-ends used by the Multi-Response Sampling module.

The DFD's external entity 'LLM Model Pool' is the `LLMClient` interface.

REAL providers (standard-library HTTPS, no SDK needed)
    openai     chat completions         key: OPENAI_API_KEY
    gemini     Google generateContent   key: GEMINI_API_KEY or GOOGLE_API_KEY
    anthropic  Claude messages          key: ANTHROPIC_API_KEY
    ollama     local server on your own machine (no key)
    ollama-cloud  Ollama's hosted API at ollama.com        key: OLLAMA_API_KEY
TESTING ONLY
    mock       offline simulator.  NOT a language model.  Every row it produces is
               flagged `simulated=True` and every report/dashboard says so.

A missing key, bad model id, rate limit or outage is reported for that model; the code
NEVER falls back to mock output.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import ssl
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass

import certifi

from .config import Config, ConfigError
from .data_collection import CLASSIFICATION, Item
from .secrets import KEY_ENV, get_key, redact


def _https_context() -> ssl.SSLContext:
    """Build a verifying context from the configured CA bundle or certifi.

    Gemini and the other hosted providers use urllib, which does not honor
    REQUESTS_CA_BUNDLE (that variable belongs to Requests). Resolve the bundle
    here so the dashboard server and CLI use the same trust configuration.
    """
    ca_bundle = (os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
                 or os.environ.get("CURL_CA_BUNDLE") or certifi.where())
    return ssl.create_default_context(cafile=ca_bundle)

# ----------------------------------------------------------------------------
# Errors (each one maps to a distinct, user-readable situation)
# ----------------------------------------------------------------------------
class LLMError(RuntimeError):
    retryable = False


class LLMUnavailable(LLMError):
    """Network failure, timeout or provider 5xx.  Retried; STD event queryFailed if all fail."""
    retryable = True


class LLMRateLimit(LLMUnavailable):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class LLMAuthError(LLMError):
    """401/403: key rejected.  Not retried."""


class LLMRequestError(LLMError):
    """400/404/422: bad model id or request.  Not retried."""


class LLMQuotaError(LLMError):
    """Out of credit / billing problem (e.g. OpenAI 'insufficient_quota').  Retrying cannot help."""


class MissingAPIKey(LLMError):
    """No key configured for the provider."""


@dataclass(frozen=True)
class GenRequest:
    item_id: str
    prompt: str
    sample_idx: int
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class Reply:
    text: str
    blocked: bool = False         # provider refused / safety-blocked the request
    finish_reason: str = ""


# ----------------------------------------------------------------------------
# Model specs
# ----------------------------------------------------------------------------
PROVIDERS = {
    "openai": "OpenAI (ChatGPT)", "gemini": "Google Gemini", "anthropic": "Anthropic Claude",
    "ollama": "Ollama (local, no key)", "ollama-cloud": "Ollama Cloud (API key)", "mock": "Mock (testing only)",
}


def parse_spec(spec: str, default_provider: str = "mock") -> tuple[str, str]:
    """'openai:gpt-x' -> ('openai', 'gpt-x').  A bare name uses the default provider."""
    spec = spec.strip()
    if ":" in spec:
        prov, _, name = spec.partition(":")
        if prov.lower() in PROVIDERS and name.strip():
            return prov.lower(), name.strip()
    return default_provider, spec


def is_simulated(spec: str, default_provider: str = "mock") -> bool:
    return parse_spec(spec, default_provider)[0] == "mock"


# ----------------------------------------------------------------------------
# HTTP helper with error classification
# ----------------------------------------------------------------------------
def http_json(url: str, headers: dict, body: dict, timeout: float, secret: str | None = None) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_https_context()) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        err_code = ""
        try:
            detail = json.loads(raw)
            detail = detail.get("error", detail)
            if isinstance(detail, dict):
                err_code = str(detail.get("code") or detail.get("type") or "")
                detail = detail.get("message", detail)
        except (json.JSONDecodeError, AttributeError):
            detail = raw[:300]
        msg = redact(f"HTTP {exc.code}: {detail}", secret)
        if exc.code in (401, 403):
            raise LLMAuthError(f"API key rejected ({msg})") from None
        if exc.code == 429 and err_code == "insufficient_quota":
            raise LLMQuotaError(f"out of credit or quota - check billing for this provider ({msg})") from None
        if exc.code == 429:
            ra = exc.headers.get("Retry-After")
            raise LLMRateLimit(f"rate limited ({msg})", float(ra) if ra and ra.replace(".", "", 1).isdigit() else None) from None
        if exc.code in (400, 404, 422):
            raise LLMRequestError(f"request rejected ({msg})") from None
        raise LLMUnavailable(f"provider error ({msg})") from None
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
        raise LLMUnavailable(redact(f"network error: {exc}", secret)) from None
    except json.JSONDecodeError:
        raise LLMUnavailable("provider returned a non-JSON response") from None


class LLMClient(ABC):
    provider = ""
    simulated = False

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, req: GenRequest) -> Reply:
        """Return one response, or raise an LLMError subclass."""


# ----------------------------------------------------------------------------
# Real providers
# ----------------------------------------------------------------------------
class _KeyedClient(LLMClient):
    base_env, base_default = "", ""

    def __init__(self, model_name: str, key: str | None, timeout: float = 90):
        super().__init__(model_name)
        if not key:
            raise MissingAPIKey(f"no API key for {PROVIDERS[self.provider]} "
                                f"(set {' or '.join(KEY_ENV[self.provider])} or enter it on the page)")
        self._key, self.timeout = key, timeout
        self.base = os.environ.get(self.base_env, self.base_default).rstrip("/")


class OpenAIClient(_KeyedClient):
    provider, base_env, base_default = "openai", "OPENAI_BASE_URL", "https://api.openai.com/v1"

    def generate(self, req: GenRequest) -> Reply:
        body = {"model": self.model_name, "max_completion_tokens": req.max_tokens,
                "messages": [{"role": "user", "content": req.prompt}], "temperature": req.temperature}
        hdr = {"Authorization": f"Bearer {self._key}"}
        try:
            data = http_json(f"{self.base}/chat/completions", hdr, body, self.timeout, self._key)
        except LLMRequestError as exc:
            if "temperature" not in str(exc).lower():
                raise
            body.pop("temperature")            # some models only accept the default temperature
            data = http_json(f"{self.base}/chat/completions", hdr, body, self.timeout, self._key)
        try:
            ch = data["choices"][0]
            msg = ch["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMUnavailable("unexpected OpenAI response format") from None
        fin = ch.get("finish_reason") or ""
        if msg.get("refusal"):
            return Reply(msg["refusal"], True, "refusal")
        return Reply(msg.get("content") or "", fin == "content_filter", fin)


class AnthropicClient(_KeyedClient):
    provider, base_env, base_default = "anthropic", "ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"

    def generate(self, req: GenRequest) -> Reply:
        body = {"model": self.model_name, "max_tokens": req.max_tokens, "temperature": min(req.temperature, 1.0),
                "messages": [{"role": "user", "content": req.prompt}]}
        hdr = {"x-api-key": self._key, "anthropic-version": "2023-06-01"}
        data = http_json(f"{self.base}/messages", hdr, body, self.timeout, self._key)
        try:
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        except (KeyError, TypeError):
            raise LLMUnavailable("unexpected Anthropic response format") from None
        stop = data.get("stop_reason") or ""
        return Reply(text, stop == "refusal", stop)


class GeminiClient(_KeyedClient):
    provider, base_env, base_default = "gemini", "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"

    def generate(self, req: GenRequest) -> Reply:
        body = {"contents": [{"role": "user", "parts": [{"text": req.prompt}]}],
                "generationConfig": {"temperature": req.temperature, "maxOutputTokens": req.max_tokens}}
        data = http_json(f"{self.base}/models/{self.model_name}:generateContent",
                         {"x-goog-api-key": self._key}, body, self.timeout, self._key)
        block = (data.get("promptFeedback") or {}).get("blockReason")
        if block:
            return Reply("", True, f"blocked:{block}")
        cands = data.get("candidates") or []
        if not cands:
            raise LLMUnavailable("Gemini returned no candidates")
        c = cands[0]
        text = "".join(p.get("text", "") for p in (c.get("content") or {}).get("parts", []))
        fin = c.get("finishReason", "")
        return Reply(text, fin in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII") and not text, fin)


class OllamaCloudClient(_KeyedClient):
    """Ollama's hosted API (ollama.com). Same request format as a local Ollama, plus a Bearer key."""
    provider, base_env, base_default = "ollama-cloud", "OLLAMA_CLOUD_URL", "https://ollama.com/api"

    def generate(self, req: GenRequest) -> Reply:
        body = {"model": self.model_name, "prompt": req.prompt, "stream": False,
                "options": {"temperature": req.temperature, "num_predict": req.max_tokens}}
        data = http_json(f"{self.base}/generate", {"Authorization": f"Bearer {self._key}"}, body, self.timeout, self._key)
        if "response" not in data:
            raise LLMUnavailable("unexpected Ollama Cloud response format")
        return Reply(data["response"])


def ollama_base() -> str:
    """Local Ollama address. Override with OLLAMA_URL (e.g. http://localhost:11434)."""
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    return url.split("/api/")[0].rstrip("/")          # also accepts the old '.../api/generate' form


def list_ollama_models(timeout: float = 4) -> list[str]:
    """Names of the models installed in the local Ollama. Raises LLMUnavailable with a how-to-fix message."""
    try:
        with urllib.request.urlopen(f"{ollama_base()}/api/tags", timeout=timeout) as r:
            return sorted(m["name"] for m in json.loads(r.read()).get("models", []))
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, ValueError, KeyError):
        raise LLMUnavailable(f"Cannot reach Ollama at {ollama_base()}. Start the Ollama app (or run `ollama serve` in a terminal), "
                             "then try again.") from None


class OllamaClient(LLMClient):
    """Local Ollama on your own machine - no key.  Friendly errors for the two common problems."""
    provider = "ollama"

    def generate(self, req: GenRequest) -> Reply:
        body = {"model": self.model_name, "prompt": req.prompt, "stream": False,
                "options": {"temperature": req.temperature, "num_predict": req.max_tokens}}
        timeout = float(os.environ.get("OLLAMA_TIMEOUT", 300))     # first request loads the model: slow on CPU
        try:
            data = http_json(f"{ollama_base()}/api/generate", {}, body, timeout)
        except LLMRequestError as exc:
            if "not found" in str(exc).lower():
                raise LLMRequestError(f"Model '{self.model_name}' is not installed in Ollama. Run `ollama pull {self.model_name}` "
                                      "(see installed models with `ollama list`).") from None
            raise
        except LLMUnavailable as exc:
            if "network error" in str(exc):
                raise LLMUnavailable(f"Cannot reach Ollama at {ollama_base()}. Start the Ollama app (or run `ollama serve`). "
                                     "If the server runs in Docker/WSL/Colab, Ollama must be reachable from there (set OLLAMA_URL).") from None
            raise
        if "response" not in data:
            raise LLMUnavailable("unexpected Ollama response format")
        return Reply(data["response"])


# ----------------------------------------------------------------------------
# Mock simulator - TESTING ONLY
# ----------------------------------------------------------------------------
def stable_int(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


class MockLLMClient(LLMClient):
    """Simulates an LLM so the pipeline can run offline and reproducibly (TESTING ONLY).

    Classification items with human votes: answers are sampled from the human distribution
    sharpened by a per-model exponent.  Open items: the reference answer (when present),
    sometimes re-phrased or replaced by another item's answer.  Items without a reference
    get a deterministic placeholder.  Output is flagged simulated everywhere it is shown.
    """
    provider, simulated = "mock", True
    GAMMAS = {"model-A": 2.5, "model-B": 1.5, "model-C": 1.0}
    OPEN_ACCURACY = {"model-A": 0.95, "model-B": 0.90, "model-C": 0.85}
    STYLES = ["{a}", "{a}.", "The answer is {a}", "I think {a}", "{A}"]

    def __init__(self, model_name: str, items: dict[str, Item], seed: int):
        super().__init__(model_name)
        self.items, self.seed = items, seed
        self.references = sorted({i.reference for i in items.values() if i.reference})

    def generate(self, req: GenRequest) -> Reply:
        rng = random.Random(stable_int(self.seed, self.model_name, req.item_id, req.sample_idx))
        item = self.items[req.item_id]
        style = rng.choice(self.STYLES)
        if item.task_type == CLASSIFICATION:
            dist = item.human_distribution()
            if dist is None:
                weights = [0.9] + [0.1 / (len(item.options) - 1)] * (len(item.options) - 1)
            else:
                g = self.GAMMAS.get(self.model_name, 1.5)
                weights = [(p + 0.02) ** g for p in dist]
            ans = rng.choices(item.options, weights=weights, k=1)[0]
        else:
            ref = item.reference or f"simulated answer {stable_int(req.item_id) % 3}"
            acc = self.OPEN_ACCURACY.get(self.model_name, 0.9)
            if rng.random() < acc:
                ans = ref
                if " " in ref and rng.random() < 0.3:
                    ans = ref.split()[-1]
            else:
                others = [r for r in self.references if r != ref] or ["unknown"]
                ans = rng.choice(others)
        return Reply(style.format(a=ans, A=ans.upper()))


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
_REAL = {"openai": OpenAIClient, "gemini": GeminiClient, "anthropic": AnthropicClient,
         "ollama-cloud": OllamaCloudClient}


def make_client(spec: str, cfg: Config, items: list[Item] | None = None) -> LLMClient:
    """Build one client or raise MissingAPIKey / ConfigError (never returns a stand-in)."""
    prov, name = parse_spec(spec, cfg.backend)
    if prov == "mock":
        return MockLLMClient(name, {i.item_id: i for i in (items or [])}, cfg.seed)
    if prov == "ollama":
        return OllamaClient(name)
    if prov in _REAL:
        return _REAL[prov](name, get_key(prov, cfg.api_keys.get(prov)))
    raise ConfigError(f"unknown provider '{prov}'")


def build_clients(cfg: Config, items: list[Item]) -> tuple[dict[str, LLMClient], dict[str, str]]:
    """-> (usable clients by spec, {spec: reason it is unavailable}).  No silent fallbacks."""
    clients: dict[str, LLMClient] = {}
    unavailable: dict[str, str] = {}
    for spec in cfg.models:
        try:
            clients[spec] = make_client(spec, cfg, items)
        except (MissingAPIKey, ConfigError) as exc:
            unavailable[spec] = redact(exc)
    return clients, unavailable
