"""Command-line entry point.

    python main.py                                   # offline demo (mock simulator, clearly labelled)
    python main.py --models openai:<model-id> gemini:<model-id> anthropic:<model-id> \
                   --data data/my.csv --prompt-template "Answer briefly: {question}" --samples 1
Keys are read from the environment or a local .env file (see .env.example).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import Config
from src.data_collection import DatasetError, Mapping, load_mapped, parse_csv_text
from src.pipeline import PipelineFailure, run_pipeline
from src.secrets import load_dotenv


def parse_args(argv=None) -> tuple[Config, Mapping | None]:
    d = Config()
    p = argparse.ArgumentParser(description="LLM disagreement & uncertainty analysis")
    p.add_argument("--data", type=Path, default=d.data_path, help="dataset CSV")
    p.add_argument("--out", type=Path, default=d.output_dir, help="output directory")
    p.add_argument("--backend", choices=["mock", "openai", "gemini", "anthropic", "ollama", "ollama-cloud"], default=d.backend,
                   help="provider for model names given without a 'provider:' prefix")
    p.add_argument("--models", nargs="+", default=d.models, help="provider:model-id, e.g. openai:<id>")
    p.add_argument("--judge", default=None, help="provider:model-id used to check whether differently worded answers agree")
    p.add_argument("--samples", type=int, default=d.n_samples, help="responses per (prompt, model)")
    p.add_argument("--temperature", type=float, default=d.temperature)
    p.add_argument("--max-tokens", type=int, default=d.max_tokens)
    p.add_argument("--threshold", type=float, default=d.agreement_threshold)
    p.add_argument("--limit", type=int, default=None, help="only first N items")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--seed", type=int, default=d.seed, help="mock backend seed")
    g = p.add_argument_group("custom dataset (any CSV)")
    g.add_argument("--prompt-template", help="e.g. 'Summarise: {article}'; {column} placeholders")
    g.add_argument("--id-column")
    g.add_argument("--task-type", choices=["open", "classification"], default="open")
    g.add_argument("--options", nargs="+", default=(), help="labels for classification")
    g.add_argument("--human-votes-column")
    g.add_argument("--reference-column")
    a = p.parse_args(argv)
    cfg = Config(data_path=a.data, output_dir=a.out, backend=a.backend, models=a.models, judge_model=a.judge,
                 n_samples=a.samples, temperature=a.temperature, max_tokens=a.max_tokens,
                 agreement_threshold=a.threshold, limit=a.limit, use_cache=not a.no_cache, seed=a.seed)
    mapping = None
    if a.prompt_template:
        mapping = Mapping(a.prompt_template, a.id_column, a.task_type, tuple(a.options),
                          a.human_votes_column, a.reference_column)
    return cfg, mapping


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    load_dotenv()
    cfg, mapping = parse_args(argv)
    try:
        items = None
        if mapping:
            cols, rows = parse_csv_text(Path(cfg.data_path).read_text(encoding="utf-8"))
            items = load_mapped(cols, rows, mapping, cfg.limit)
        result = run_pipeline(cfg, items)
    except (DatasetError, OSError) as exc:
        logging.error("Invalid input: %s", exc)
        return 2
    except PipelineFailure as exc:
        logging.error("%s: %s", exc.kind, exc.message)
        return {"invalid_input": 2, "no_models": 3}.get(exc.kind, 1)
    print("\n=== FINDINGS ===")
    for line in result["comparison"]["findings"]:
        print(" -", line)
    print("\n=== SUMMARY ===")
    print(json.dumps(result["summary"], indent=2, default=str))
    print(f"\nFull report: {result['report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
