"""End-to-end pipeline.  Every move between stages goes through `StateMachine.fire`, which
validates it against the State Transition Diagram and records it (see src/states.py)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from .comparison import assign_clusters, compare_models
from .config import Config, ConfigError
from .data_collection import DatasetError, Item, load_dataset
from .disagreement import analyze_all, summarize
from .explain import explain_results
from .llm_clients import build_clients, make_client, MissingAPIKey, parse_spec
from .normalization import normalize_responses
from .report_html import build_html_report
from .reporting import generate_report, write_comparison_exports
from .sampling import sample_responses
from .secrets import redact
from .states import StateMachine, diagram_model, render_svg, to_drawio

log = logging.getLogger("pipeline")
Progress = Callable[[str, float, str], None]        # (stage, fraction 0..1 of the whole run, message)


class PipelineFailure(RuntimeError):
    """Run stopped.  kind: invalid_input | no_models | internal.  Carries the state trace."""

    def __init__(self, kind: str, message: str, machine: StateMachine, details: dict | None = None):
        super().__init__(message)
        self.kind, self.message, self.machine, self.details = kind, message, machine, details or {}


def _go(sm: StateMachine, event: str, guard: str = "", note: str = "") -> None:
    before = sm.state.value
    sm.fire(event, guard, note)
    log.info("[STATE] %s -> %s  (%s%s)", before, sm.state.value, event, f": {note}" if note else "")


def run_pipeline(cfg: Config, items: list[Item] | None = None, progress: Progress | None = None,
                 judge_client=None) -> dict:
    """Run all stages. `items` may be supplied (custom prompts/datasets); otherwise cfg.data_path is loaded.
    `judge_client` lets tests inject a stand-in judge; normally it is built from cfg.judge_model."""
    sm = StateMachine()
    emit = progress or (lambda *_: None)
    _go(sm, "start")
    _go(sm, "submitInput")
    emit("configuring", 0.02, "validating input")

    # ---------------- CONFIGURING ----------------
    judge = judge_client
    try:
        cfg.validate()
        items = items if items is not None else load_dataset(cfg.data_path, cfg.limit)
        if not items:
            raise DatasetError("no items to analyse")
        n_req = len(items) * len(cfg.models) * cfg.n_samples
        if n_req > cfg.max_requests:
            raise ConfigError(f"{n_req} requests requested; the limit is {cfg.max_requests}. "
                              "Reduce items, models or samples.")
        if cfg.judge_model and judge is None:
            if parse_spec(cfg.judge_model, cfg.backend)[0] == "mock":
                raise ConfigError("the judge model must be a real model, not the mock simulator")
            try:
                judge = make_client(cfg.judge_model, cfg)
            except MissingAPIKey as exc:
                raise ConfigError(f"judge model: {exc}") from None
    except (DatasetError, ConfigError) as exc:
        msg = redact(exc)
        _go(sm, "configDone", "dataset or settings invalid", msg[:300])
        raise PipelineFailure("invalid_input", msg, sm) from None
    _go(sm, "configDone", "dataset and settings valid", f"{len(items)} items")
    emit("ready", 0.05, f"{len(items)} items ready")

    try:
        # ---------------- QUERYING ----------------
        _go(sm, "startAnalysis")
        clients, unavailable = build_clients(cfg, items)
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache = out_dir / "response_cache.jsonl"

        def samp_progress(done, total, msg):
            emit("querying", 0.05 + 0.55 * done / max(total, 1), msg)

        if clients:
            raw, report = sample_responses(items, clients, cfg, cache, samp_progress)
        else:
            import pandas as pd
            raw = pd.DataFrame(columns=["item_id", "model", "sample_idx", "raw_text", "status", "error", "simulated"])
            report = {}
        for spec, reason in unavailable.items():
            report[spec] = {"ok": 0, "blocked": 0, "errors": 0, "disabled_reason": reason, "simulated": False}
        usable = {m: r for m, r in report.items() if r["ok"] + r["blocked"] > 0}
        failed = {m: r["disabled_reason"] or f"{r['errors']} failed requests"
                  for m, r in report.items() if m not in usable or r["disabled_reason"] or r["errors"]}
        if not usable:
            reason = "; ".join(f"{m}: {why}" for m, why in failed.items()) or "no responses were received"
            _go(sm, "queryFailed", "no model returned usable responses", reason[:300])
            raise PipelineFailure("no_models", reason, sm, {"model_report": report})
        _go(sm, "responsesReceived", "at least one model returned usable responses",
            f"{len(usable)}/{len(cfg.models)} models usable")

        # ---------------- NORMALIZING ----------------
        emit("normalizing", 0.62, "normalizing responses")
        responses = normalize_responses(raw, items)
        _go(sm, "normalizationDone")

        # ---------------- EVALUATING (semantic equivalence) ----------------
        emit("evaluating", 0.65, "comparing answers" + (" with judge model" if judge else ""))
        responses, stage = assign_clusters(
            items, responses, cfg, judge,
            lambda d, t, msg: emit("evaluating", 0.65 + 0.15 * d / max(t, 1), msg))
        _go(sm, "evaluationDone", note=("judge model used" if judge else "lexical checks only (no judge model)"))

        # ---------------- DETECTING ----------------
        emit("detecting", 0.82, "computing disagreement metrics")
        metrics = analyze_all(items, responses, cfg)
        summary = summarize(metrics, cfg)
        comparison = compare_models(items, responses, cfg, stage)
        simulated = bool(responses["simulated"].any()) if len(responses) else False
        comparison["simulated"] = simulated
        comparison["model_report"] = report
        if simulated:
            comparison["findings"].insert(0, "SIMULATED RUN: at least one model is the mock simulator, so these "
                                             "numbers validate the pipeline only and say nothing about real LLMs.")
        summary["comparison"] = comparison["overall"]
        comparison["explain"] = explain_results(comparison, summary)
        summary["simulated"] = simulated
        disagree = comparison["overall"]["disagree_n"] > 0 or summary["n_flagged"] > 0
        _go(sm, "metricsComputed",
            "disagreement detected, debate not enabled" if disagree else "no disagreement detected")

        # ---------------- RESULTS_STORED -> GENERATING_REPORT -> REPORT_DELIVERED ----------------
        emit("reporting", 0.9, "saving results")
        _go(sm, "resultsReady")
        report_path = generate_report(items, responses, metrics, summary, cfg, simulated)
        _go(sm, "reportCompiled")
        trace = sm.to_dict()
        svg = render_svg(sm)
        html = build_html_report(comparison, summary, {"models": cfg.models, "samples": cfg.n_samples,
                                                       "temperature": cfg.temperature, "judge": cfg.judge_model},
                                 svg, trace["trace"], out_dir.name)
        write_comparison_exports(out_dir, comparison, trace, svg, to_drawio(sm), html)
        emit("done", 1.0, "finished")
    except PipelineFailure:
        raise
    except Exception as exc:                                    # unexpected bug: keep the trace
        sm.fail(redact(exc))
        raise PipelineFailure("internal", f"unexpected error in {sm.state.value}: {redact(exc)}", sm) from exc

    return {"items": items, "responses": responses, "metrics": metrics, "summary": summary,
            "comparison": comparison, "report": report_path, "machine": sm,
            "diagram": diagram_model(sm), "model_report": report}
