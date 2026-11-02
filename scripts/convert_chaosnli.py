"""Convert ChaosNLI (SNLI / MNLI part) JSONL into this project's CSV format.

Field names follow the ChaosNLI README: uid, label_counter {e,n,c},
example {premise, hypothesis}.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

LABELS = {"e": "entailment", "n": "neutral", "c": "contradiction"}


def convert(src: Path, dst: Path, n: int | None, seed: int) -> int:
    records = []
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"{src} contains no records")
    if n and n < len(records):
        records = random.Random(seed).sample(records, n)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["id", "task_type", "instruction", "text", "options", "human_votes", "reference"])
        for r in records:
            try:
                ex, counter = r["example"], r["label_counter"]
                text = f"Premise: {ex['premise']} Hypothesis: {ex['hypothesis']}"
                votes = "|".join(f"{LABELS[k]}:{int(v)}" for k, v in counter.items() if int(v) > 0)
            except KeyError as exc:
                raise SystemExit(f"unexpected record format (missing {exc}); is this an SNLI/MNLI file?")
            w.writerow([r["uid"], "classification", "Does the premise entail the hypothesis?",
                        text, "entailment|neutral|contradiction", votes, ""])
    return len(records)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n", type=int, default=None, help="random subset size")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"wrote {convert(a.input, a.output, a.n, a.seed)} items to {a.output}")
    sys.exit(0)
