"""Web server (standard library only).   python web/server.py [port]   ->  http://127.0.0.1:8000

GET  /                       the page
GET  /api/providers          which providers have a key configured on the server (booleans only)
GET  /api/sample             the built-in sample dataset
GET  /api/example/<name>      example CSVs to try (whitelisted names)
GET  /api/ollama/models      models installed in your local Ollama (or how to fix the connection)
GET  /api/std                the State Transition Diagram (SVG) drawn from the code's own table
POST /api/inspect            {csv}  -> columns, row count, preview (for column mapping)
POST /api/test-model         {model, key?} -> sends one tiny request to check key + model id
POST /api/run                {...} -> {job_id}; the analysis runs in a background thread
GET  /api/job/<id>           progress, live state trace, and (when finished) the full result
GET  /api/export/<run>/<file>  download a result file of a saved run
GET  /api/runs               saved runs
Keys typed in the page are used for that job only: never stored, logged, cached or echoed back.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import Config, ConfigError                          # noqa: E402
from src.data_collection import (CLASSIFICATION, DatasetError, Mapping, inspect_csv, items_from_prompts,  # noqa: E402
                                 load_dataset, load_mapped, parse_csv_text)
from src.llm_clients import (GenRequest, LLMError, PROVIDERS, list_ollama_models, make_client,  # noqa: E402
                             parse_spec)
from src.pipeline import PipelineFailure, run_pipeline                # noqa: E402
from src.secrets import key_status, load_dotenv, redact              # noqa: E402
from src.states import StateMachine, diagram_model, render_svg      # noqa: E402

RUNS = ROOT / "runs"
MAX_BODY = 4_000_000
EXPORTS = {"responses.csv": "raw_responses.csv", "items.csv": "items_comparison.csv",
           "pairwise.csv": "pairwise_disagreement.csv", "pair_verdicts.csv": "answer_pair_verdicts.csv",
           "comparison.json": "comparison.json", "summary.json": "summary.json", "report.md": "report.md",
           "report.html": "report.html",
           "trace.json": "state_trace.json", "std.svg": "std.svg", "std.drawio": "std.drawio",
           "config.json": "config.json"}
EXAMPLES = {"open_questions.csv": "open_questions.csv", "sentiment_classification.csv": "sentiment_classification.csv",
            "prompts_only.csv": "prompts_only.csv"}
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


# ----------------------------------------------------------------------------
# Request -> Config / items  (kept separate so it can be tested without HTTP)
# ----------------------------------------------------------------------------
def build_job_inputs(p: dict, out_dir: Path) -> tuple[Config, list]:
    mode = p.get("data_mode", "sample")
    models = [m.strip() for m in p.get("models", []) if str(m).strip()][:8]
    if not models:
        raise ConfigError("select at least one model")
    keys = {k: v for k, v in (p.get("keys") or {}).items() if isinstance(v, str) and v.strip()}
    try:
        cfg = Config(output_dir=out_dir, models=models, backend="mock", api_keys=keys,
                     n_samples=int(p.get("samples", 1)), temperature=float(p.get("temperature", 0.7)),
                     max_tokens=int(p.get("max_tokens", 512)), judge_model=(p.get("judge") or None),
                     agreement_threshold=float(p.get("threshold", 0.75)), use_cache=bool(p.get("cache", False)))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"bad parameters: {exc}") from None
    m = p.get("mapping") or {}
    task = m.get("task_type", "open")
    options = tuple(o.strip() for o in str(m.get("options", "")).split(",") if o.strip())
    if mode == "sample":
        return cfg, load_dataset(ROOT / "data" / "sample_dataset.csv")
    if mode == "prompts":
        raw = str(p.get("prompts", "")).strip()
        lines = raw.split("\n\n") if "\n\n" in raw else raw.split("\n")   # blank-line blocks, else one per line
        return cfg, items_from_prompts(lines, task, options)
    if mode == "csv":
        cols, rows = parse_csv_text(str(p.get("csv", "")))
        mp = Mapping(m.get("prompt_template", ""), m.get("id_column") or None, task, options,
                     m.get("human_votes_column") or None, m.get("reference_column") or None)
        return cfg, load_mapped(cols, rows, mp)
    raise DatasetError(f"unknown data mode '{mode}'")


def _job_worker(job_id: str, payload: dict) -> None:
    job = JOBS[job_id]
    run_dir = RUNS / job["run_id"]

    def progress(stage, frac, msg):
        with LOCK:
            job.update(stage=stage, progress=round(frac, 3), message=msg)

    try:
        cfg, items = build_job_inputs(payload, run_dir)
        res = run_pipeline(cfg, items, progress)
        text = {i.item_id: i.text for i in res["items"]}
        mrows = res["metrics"].astype(object).where(res["metrics"].notna(), None).to_dict("records")
        for r in mrows:
            r["text"] = text[r["item_id"]]
        result = {"summary": res["summary"], "comparison": res["comparison"], "item_metrics": mrows,
                  "diagram": res["diagram"], "svg": render_svg(res["machine"]), "run_id": job["run_id"],
                  "config": {"models": cfg.models, "samples": cfg.n_samples, "judge": cfg.judge_model,
                             "temperature": cfg.temperature}}
        with LOCK:
            job.update(status="done", result=result, progress=1.0, stage="done")
    except PipelineFailure as exc:
        with LOCK:
            job.update(status="failed", error={"kind": exc.kind, "message": exc.message, "details": exc.details},
                       svg=render_svg(exc.machine), diagram=diagram_model(exc.machine))
    except (DatasetError, ConfigError) as exc:                   # failed before the pipeline started
        sm = StateMachine()
        sm.fire("start"); sm.fire("submitInput"); sm.fire("configDone", "dataset or settings invalid", str(exc)[:300])
        with LOCK:
            job.update(status="failed", error={"kind": "invalid_input", "message": redact(exc)},
                       svg=render_svg(sm), diagram=diagram_model(sm))
    except Exception as exc:                                     # never leave a job hanging
        logging.exception("job crashed")
        with LOCK:
            job.update(status="failed", error={"kind": "internal", "message": redact(exc)})


def start_job(payload: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    run_id = time.strftime("%Y%m%d-%H%M%S-") + job_id[:4]
    JOBS[job_id] = {"status": "running", "progress": 0.0, "stage": "queued", "message": "", "run_id": run_id}
    threading.Thread(target=_job_worker, args=(job_id, payload), daemon=True).start()
    return job_id


def test_model(spec: str, key: str | None) -> dict:
    """One tiny real request: verifies key + model id.  Never returns the key."""
    prov, _ = parse_spec(spec, "mock")
    if prov == "mock":
        return {"ok": True, "simulated": True, "message": "Mock simulator (testing only) - no API was contacted."}
    try:
        cfg = Config(api_keys={prov: key} if key else {}, models=[spec])
        reply = make_client(spec, cfg).generate(GenRequest("test", "Reply with the single word: OK", 0, 0.0, 16))
        return {"ok": True, "message": f"Connected. Model replied: {reply.text.strip()[:60]!r}"}
    except LLMError as exc:
        return {"ok": False, "message": redact(exc, key)}
    except ConfigError as exc:
        return {"ok": False, "message": redact(exc, key)}


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode(),
                   "application/json")

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            self._json(413, {"error": "upload too large (max 4 MB)"})
            return None
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "request body is not valid JSON"})
            return None

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, (Path(__file__).parent / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/providers":
            self._json(200, {"providers": PROVIDERS, "server_keys": key_status()})
        elif path.startswith("/api/example/"):
            name = path.rsplit("/", 1)[1]
            f = ROOT / "data" / "examples" / EXAMPLES[name] if name in EXAMPLES else None
            if not f or not f.is_file():
                return self._json(404, {"error": "unknown example"})
            self._send(200, f.read_bytes(), "text/csv; charset=utf-8", {"Content-Disposition": f'attachment; filename="{name}"'})
        elif path == "/api/std":
            self._send(200, render_svg(None).encode(), "image/svg+xml")
        elif path == "/api/ollama/models":
            try:
                self._json(200, {"ok": True, "models": list_ollama_models()})
            except LLMError as exc:
                self._json(200, {"ok": False, "models": [], "message": str(exc)})
        elif path == "/api/sample":
            self._send(200, (ROOT / "data" / "sample_dataset.csv").read_bytes(), "text/csv; charset=utf-8")
        elif path.startswith("/api/job/"):
            job = JOBS.get(path.rsplit("/", 1)[1])
            if not job:
                return self._json(404, {"error": "unknown job"})
            with LOCK:
                snap = dict(job)
            self._json(200, snap)
        elif path == "/api/runs":
            runs = sorted((d.name for d in RUNS.glob("*") if d.is_dir()), reverse=True)[:50] if RUNS.exists() else []
            self._json(200, {"runs": runs})
        elif path.startswith("/api/export/"):
            parts = path.split("/")
            if len(parts) != 5 or not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{4}", parts[3]) or parts[4] not in EXPORTS:
                return self._json(404, {"error": "not found"})
            f = RUNS / parts[3] / EXPORTS[parts[4]]
            if not f.is_file():
                return self._json(404, {"error": "file not found"})
            self._send(200, f.read_bytes(), "application/octet-stream",
                       {"Content-Disposition": f'attachment; filename="{parts[3]}_{parts[4]}"'})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        p = self._body()
        if p is None:
            return
        if self.path == "/api/inspect":
            try:
                self._json(200, inspect_csv(str(p.get("csv", ""))))
            except DatasetError as exc:
                self._json(400, {"error": str(exc)})
        elif self.path == "/api/test-model":
            self._json(200, test_model(str(p.get("model", "")), (p.get("key") or None)))
        elif self.path == "/api/run":
            self._json(202, {"job_id": start_job(p)})
        else:
            self._json(404, {"error": "not found"})

    def log_message(self, fmt, *args):          # request lines only; bodies (which may hold keys) are never logged
        logging.info("%s %s", self.address_string(), fmt % args)


def make_server(port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    load_dotenv()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Serving on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    make_server(port).serve_forever()
