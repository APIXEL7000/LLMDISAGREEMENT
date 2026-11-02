"""Multi-Response Sampling Module  (DFD Process P2, STD state QUERYING).

Every prompt is sent to every available model `n_samples` times.  Repeating the query
exposes a model's *internal* uncertainty (Nan et al. 2025; Xiong et al. 2024; Wang et al. 2022).

Failure handling (nothing is ever replaced by simulated output):
* a request is retried with exponential back-off (rate limits honour Retry-After);
* a rejected key / unknown model id disables that model immediately, with the reason;
* 3 consecutive requests that still fail disable the model;
* other models keep running; the report states which models failed and why.

Responses are cached in a JSONL file (text only, never keys) so interrupted runs resume.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import Config
from .data_collection import Item, build_prompt
from .llm_clients import GenRequest, LLMClient, LLMError, LLMRateLimit, Reply
from .secrets import redact

log = logging.getLogger(__name__)
Progress = Callable[[int, int, str], None]
MAX_CONSECUTIVE_FAILURES = 3
MAX_WAIT = 20.0


def _cache_key(model: str, req: GenRequest) -> str:
    raw = f"{model}|{req.item_id}|{req.sample_idx}|{req.temperature}|{req.max_tokens}|{req.prompt}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if path.is_file():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    cache[rec["key"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue            # ignore a half-written last line
    return cache


def _generate_with_retry(client: LLMClient, req: GenRequest, cfg: Config) -> Reply:
    for attempt in range(1, cfg.max_retries + 1):
        try:
            return client.generate(req)
        except LLMError as exc:
            if not exc.retryable or attempt == cfg.max_retries:
                raise
            wait = cfg.retry_base_delay * 2 ** (attempt - 1)
            if isinstance(exc, LLMRateLimit) and exc.retry_after:
                wait = exc.retry_after
            wait = min(wait, MAX_WAIT)
            log.warning("%s attempt %d/%d failed (%s); retrying in %.1fs", client.model_name, attempt,
                        cfg.max_retries, exc, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def sample_responses(items: list[Item], clients: dict[str, LLMClient], cfg: Config, cache_path: Path,
                     progress: Progress | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (responses DataFrame, per-model report).

    DataFrame columns: item_id, model, sample_idx, raw_text, status (ok|blocked|error),
    error, simulated.   Report: {model: {ok, blocked, errors, disabled_reason, simulated}}.
    """
    cache = _load_cache(cache_path) if cfg.use_cache else {}
    report = {m: {"ok": 0, "blocked": 0, "errors": 0, "disabled_reason": None,
                  "simulated": c.simulated} for m, c in clients.items()}
    streak = {m: 0 for m in clients}
    total = len(items) * len(clients) * cfg.n_samples
    rows, done = [], 0
    if cfg.use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(cache_path, "a", encoding="utf-8") if cfg.use_cache else None
    try:
        for item in items:
            prompt = build_prompt(item)
            for model, client in clients.items():
                for k in range(cfg.n_samples):
                    done += 1
                    if report[model]["disabled_reason"]:
                        continue
                    req = GenRequest(item.item_id, prompt, k, cfg.temperature, cfg.max_tokens)
                    key = _cache_key(model, req)
                    base = {"item_id": item.item_id, "model": model, "sample_idx": k,
                            "simulated": client.simulated}
                    if key in cache:
                        rec = cache[key]
                        rows.append({**base, "raw_text": rec["raw_text"], "status": rec["status"], "error": ""})
                        report[model]["blocked" if rec["status"] == "blocked" else "ok"] += 1
                    else:
                        try:
                            reply = _generate_with_retry(client, req, cfg)
                        except LLMError as exc:
                            msg = redact(exc)
                            rows.append({**base, "raw_text": "", "status": "error", "error": msg})
                            report[model]["errors"] += 1
                            streak[model] += 1
                            if not exc.retryable:
                                report[model]["disabled_reason"] = msg
                            elif streak[model] >= MAX_CONSECUTIVE_FAILURES:
                                report[model]["disabled_reason"] = f"{streak[model]} consecutive failures; last: {msg}"
                        else:
                            streak[model] = 0
                            status = "blocked" if reply.blocked else "ok"
                            rows.append({**base, "raw_text": reply.text, "status": status, "error": ""})
                            report[model]["blocked" if reply.blocked else "ok"] += 1
                            if fh:
                                fh.write(json.dumps({"key": key, "raw_text": reply.text, "status": status}) + "\n")
                                fh.flush()
                    if progress and (done % max(1, total // 200) == 0 or done == total):
                        progress(done, total, f"{model} - item {item.item_id}")
    finally:
        if fh:
            fh.close()
    df = pd.DataFrame(rows, columns=["item_id", "model", "sample_idx", "raw_text", "status", "error", "simulated"])
    log.info("sampling finished: %s", {m: {k: v for k, v in r.items() if k != "simulated"} for m, r in report.items()})
    return df, report
