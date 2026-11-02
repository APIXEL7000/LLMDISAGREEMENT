"""Reporting Module - basic version  (DFD Process P4, STD GENERATING_REPORT).

Writes the final results and highlights uncertain cases:
  item_metrics.csv, flagged_cases.csv, raw_responses.csv, summary.json,
  report.md, agreement_hist.png, llm_vs_human_entropy.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # no display needed
import matplotlib.pyplot as plt
import pandas as pd

from .config import Config
from .data_collection import Item


def _plots(m: pd.DataFrame, cfg: Config, out: Path) -> list[Path]:
    paths = []
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(m["agreement"].dropna(), bins=10, range=(0, 1), edgecolor="black")
    ax.axvline(cfg.agreement_threshold, color="red", linestyle="--",
               label=f"threshold = {cfg.agreement_threshold}")
    ax.set(xlabel="Agreement (share of majority answer)", ylabel="Number of items",
           title="Distribution of LLM agreement")
    ax.legend(); fig.tight_layout()
    paths.append(out / "agreement_hist.png"); fig.savefig(paths[-1], dpi=150); plt.close(fig)

    if "human_entropy_norm" in m and m["human_entropy_norm"].notna().any():
        h = m[m["human_entropy_norm"].notna()]
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(h["human_entropy_norm"], h["entropy_norm"], c=h["overconfident"].map(
            {True: "red", False: "tab:blue"}), edgecolor="black")
        ax.plot([0, 1], [0, 1], "k:", linewidth=1)
        ax.set(xlabel="Human label entropy (normalised)", ylabel="LLM answer entropy (normalised)",
               title="LLM vs human disagreement (red = overconfident)",
               xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
        fig.tight_layout()
        paths.append(out / "llm_vs_human_entropy.png"); fig.savefig(paths[-1], dpi=150); plt.close(fig)
    return paths


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def generate_report(items: list[Item], responses: pd.DataFrame, metrics: pd.DataFrame,
                    summary: dict, cfg: Config, simulated: bool = False) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    responses.to_csv(out / "raw_responses.csv", index=False)
    metrics.to_csv(out / "item_metrics.csv", index=False)
    flagged = metrics[metrics["disagreement"]].sort_values("agreement")
    flagged.to_csv(out / "flagged_cases.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _plots(metrics, cfg, out)
    (out / "config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items() if k != "api_keys"}, indent=2))

    text = {i.item_id: i.text for i in items}
    top = flagged.head(cfg.top_k_uncertain).copy()
    top["text"] = top["item_id"].map(lambda i: text[i][:70] + ("..." if len(text[i]) > 70 else ""))
    show = [c for c in ["item_id", "text", "majority_answer", "agreement", "entropy_norm",
                        "cross_model_disagree", "human_agreement", "overconfident"] if c in top]

    md = ["# Disagreement Analysis Report", ""]
    if simulated:
        md += ["> **SIMULATED RUN.** At least one model is the mock simulator, not a language model. "
               "These numbers validate the pipeline only and are **not** evidence about real LLMs.", ""]
    md += ["## Run configuration",
           f"- models: {', '.join(cfg.models)}",
           f"- samples per (prompt, model): {cfg.n_samples}  |  temperature: {cfg.temperature}",
           f"- flag item when agreement < {cfg.agreement_threshold}", "",
           "## Summary", "```json", json.dumps(summary, indent=2), "```", "",
           f"## Uncertain cases (top {len(top)} of {len(flagged)} flagged, lowest agreement first)",
           _md_table(top[show]) if len(top) else "_No item was flagged._", "",
           "## Files", "item_metrics.csv, flagged_cases.csv, raw_responses.csv, summary.json, "
           "agreement_hist.png, llm_vs_human_entropy.png"]
    path = out / "report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def write_comparison_exports(out: Path, comparison: dict, trace: dict, svg: str, drawio: str, html: str = "") -> None:
    """Cross-model results + the run's state trace and diagrams (all key-free)."""
    out = Path(out)
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2, default=str))
    flat = []
    for r in comparison["items"]:
        row = {"item_id": r["item_id"], "category": r["category"], "group": r["group"],
               "dissenting_models": ";".join(r["dissenting_models"]), "prompt": r["prompt"]}
        for m, a in r["answers"].items():
            row[f"{m} | response"] = a["text"] if a["status"] != "error" else f"[ERROR] {a['error']}"
        flat.append(row)
    pd.DataFrame(flat).to_csv(out / "items_comparison.csv", index=False)
    pd.DataFrame(comparison["pairs"]).to_csv(out / "pairwise_disagreement.csv", index=False)
    pd.DataFrame(comparison["pair_log"]).to_csv(out / "answer_pair_verdicts.csv", index=False)
    (out / "state_trace.json").write_text(json.dumps(trace, indent=2))
    (out / "std.svg").write_text(svg, encoding="utf-8")
    (out / "std.drawio").write_text(drawio, encoding="utf-8")
    if html:
        (out / "report.html").write_text(html, encoding="utf-8")
