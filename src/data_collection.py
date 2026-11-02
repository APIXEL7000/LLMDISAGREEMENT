"""Data Collection Module  (DFD Process P1, STD states CONFIGURING -> DATASET_READY).

Loads the evaluation dataset from CSV, validates it, and prepares one prompt
per item.  Items may carry an optional distribution of human votes; this is what
the literature (Lee et al. 2023; Ni et al. 2025) compares LLM output against.

CSV columns
-----------
id           unique item id                                   (required)
task_type    'classification' or 'open'                       (required)
instruction  task instruction shown to the LLM                (required)
text         the content to judge / the question              (required)
options      'a|b|c' label set (classification only)
human_votes  'a:7|b:3' annotator vote counts (optional)
reference    reference answer (open items; used by the mock backend now and by
             the LLM-as-Judge module in the remaining 50%)
"""
from __future__ import annotations

import csv
import io
import string
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CLASSIFICATION = "classification"
OPEN = "open"
REQUIRED_COLUMNS = ["id", "task_type", "instruction", "text"]


class DatasetError(ValueError):
    """Raised when the dataset is invalid (STD guard: [dataset invalid])."""


@dataclass(frozen=True)
class Item:
    item_id: str
    task_type: str
    instruction: str
    text: str
    options: tuple[str, ...] = ()
    human_votes: dict[str, int] | None = None
    reference: str | None = None
    prompt: str | None = None           # fully rendered prompt (custom datasets / typed prompts)

    def human_distribution(self) -> np.ndarray | None:
        """Human vote shares in `options` order, or None if no votes."""
        if not self.human_votes:
            return None
        counts = np.array([self.human_votes.get(o, 0) for o in self.options], dtype=float)
        return counts / counts.sum()


def parse_options(raw: str) -> tuple[str, ...]:
    return tuple(o.strip() for o in raw.split("|") if o.strip())


def parse_votes(raw: str) -> dict[str, int]:
    """'entailment:7|neutral:3' -> {'entailment': 7, 'neutral': 3}."""
    votes: dict[str, int] = {}
    for part in raw.split("|"):
        part = part.strip()
        if not part:
            continue
        label, sep, count = part.rpartition(":")
        if not sep or not label.strip():
            raise ValueError(f"bad vote entry '{part}' (expected label:count)")
        votes[label.strip()] = int(count)
    return votes


def load_dataset(path: Path, limit: int | None = None) -> list[Item]:
    """Load and validate the dataset. Collects *all* problems before raising."""
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"dataset file not found: {path}")
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise DatasetError(f"missing required column(s): {missing}")
        rows = list(reader)
    if not rows:
        raise DatasetError("dataset is empty")

    items: list[Item] = []
    errors: list[str] = []
    seen: set[str] = set()
    for line_no, row in enumerate(rows, start=2):   # line 1 is the header
        rid = (row.get("id") or "").strip()
        where = f"line {line_no} (id='{rid}')"
        if not rid:
            errors.append(f"line {line_no}: empty id"); continue
        if rid in seen:
            errors.append(f"{where}: duplicate id")
        seen.add(rid)

        ttype = (row.get("task_type") or "").strip().lower()
        if ttype not in (CLASSIFICATION, OPEN):
            errors.append(f"{where}: task_type must be 'classification' or 'open'"); continue
        if not (row.get("text") or "").strip():
            errors.append(f"{where}: empty text")
        if not (row.get("instruction") or "").strip():
            errors.append(f"{where}: empty instruction")

        options = parse_options(row.get("options") or "")
        if ttype == CLASSIFICATION and len(options) < 2:
            errors.append(f"{where}: classification items need >= 2 options")

        votes = None
        if (row.get("human_votes") or "").strip():
            try:
                votes = parse_votes(row["human_votes"])
            except ValueError as exc:
                errors.append(f"{where}: {exc}")
            else:
                if ttype != CLASSIFICATION:
                    errors.append(f"{where}: human_votes only supported for classification")
                elif not set(votes) <= set(options):
                    errors.append(f"{where}: human_votes labels {sorted(set(votes) - set(options))} not in options")
                elif sum(votes.values()) <= 0 or min(votes.values()) < 0:
                    errors.append(f"{where}: human_votes counts must be >= 0 and sum > 0")

        items.append(Item(
            item_id=rid, task_type=ttype,
            instruction=(row.get("instruction") or "").strip(),
            text=(row.get("text") or "").strip(),
            options=options, human_votes=votes,
            reference=(row.get("reference") or "").strip() or None,
        ))
    if errors:
        raise DatasetError("; ".join(errors))
    return items[:limit] if limit else items


def build_prompt(item: Item) -> str:
    """Prepare the LLM prompt for one item (paper: 'prepares prompts')."""
    if item.prompt:                                   # user-defined prompt: sent as written
        if item.task_type == CLASSIFICATION:
            return (f"{item.prompt}\n\nAnswer with exactly one of: {', '.join(item.options)}. "
                    f"Reply with only that label.")
        return item.prompt
    if item.task_type == CLASSIFICATION:
        return (f"{item.instruction}\n\n{item.text}\n\n"
                f"Answer with exactly one of: {', '.join(item.options)}. "
                f"Reply with only that label.")
    return f"{item.instruction}\n\n{item.text}\n\nAnswer concisely in a few words."


# ----------------------------------------------------------------------------
# Custom datasets: any CSV + column mapping + prompt template
# ----------------------------------------------------------------------------
MAX_ROWS = 2000


@dataclass
class Mapping:
    """How to turn an arbitrary CSV into items.  Nothing here is tied to a specific format."""
    prompt_template: str                    # e.g. "Summarise: {article}"  ({column} placeholders)
    id_column: str | None = None            # None -> row number
    task_type: str = OPEN                   # 'open' (free text) or 'classification'
    options: tuple[str, ...] = ()           # classification label set
    human_votes_column: str | None = None   # optional 'label:count|label:count'
    reference_column: str | None = None     # optional reference answer (shown, kept for later judging)


def parse_csv_text(text: str) -> tuple[list[str], list[dict]]:
    """Parse CSV text (delimiter auto-detected among , ; tab). Returns (columns, rows)."""
    text = text.lstrip("\ufeff")
    if not text.strip():
        raise DatasetError("the file is empty")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text, newline=""), dialect=dialect)
    cols = [c.strip() for c in (reader.fieldnames or []) if c is not None and c.strip()]
    if not cols:
        raise DatasetError("could not find a header row")
    if len(set(cols)) != len(cols):
        raise DatasetError("duplicate column names in the header")
    rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items() if k is not None} for r in reader]
    rows = [r for r in rows if any(r.values())]
    if not rows:
        raise DatasetError("the file has a header but no data rows")
    if len(rows) > MAX_ROWS:
        raise DatasetError(f"too many rows ({len(rows)}); the limit is {MAX_ROWS}. Use a subset.")
    return cols, rows


def inspect_csv(text: str) -> dict:
    """Used by the web page to offer column mapping: columns, row count, preview."""
    cols, rows = parse_csv_text(text)
    return {"columns": cols, "n_rows": len(rows), "preview": rows[:5]}


def template_fields(template: str) -> list[str]:
    try:
        return [f for _, f, _, _ in string.Formatter().parse(template) if f]
    except ValueError as exc:
        raise DatasetError(f"prompt template is malformed ({exc}); use {{{{ and }}}} for literal braces") from None


def load_mapped(columns: list[str], rows: list[dict], mapping: Mapping, limit: int | None = None) -> list[Item]:
    """Build items from parsed rows according to `mapping`. Reports every problem at once."""
    errors: list[str] = []
    tmpl = (mapping.prompt_template or "").strip()
    if not tmpl:
        errors.append("the prompt template is empty")
    unknown = [f for f in template_fields(tmpl) if f not in columns]
    if unknown:
        errors.append(f"prompt template uses unknown column(s) {unknown}; available: {columns}")
    for label, col in (("id column", mapping.id_column), ("human votes column", mapping.human_votes_column),
                       ("reference column", mapping.reference_column)):
        if col and col not in columns:
            errors.append(f"{label} '{col}' is not in the file")
    if mapping.task_type not in (CLASSIFICATION, OPEN):
        errors.append("task type must be 'open' or 'classification'")
    if mapping.task_type == CLASSIFICATION and len(mapping.options) < 2:
        errors.append("classification needs at least 2 options")
    if errors:
        raise DatasetError("; ".join(errors))

    items: list[Item] = []
    seen: set[str] = set()
    for n, row in enumerate(rows, start=1):
        rid = (row.get(mapping.id_column, "") if mapping.id_column else "") or str(n)
        if rid in seen:
            errors.append(f"row {n}: duplicate id '{rid}'")
        seen.add(rid)
        prompt = tmpl.format_map({c: row.get(c, "") for c in columns}).strip()
        if not prompt:
            errors.append(f"row {n}: the rendered prompt is empty")
        votes = None
        if mapping.human_votes_column and row.get(mapping.human_votes_column):
            try:
                votes = parse_votes(row[mapping.human_votes_column])
                if not set(votes) <= set(mapping.options):
                    errors.append(f"row {n}: human votes use labels not in the options")
            except ValueError as exc:
                errors.append(f"row {n}: {exc}")
        items.append(Item(item_id=rid, task_type=mapping.task_type, instruction="", text=prompt,
                          options=tuple(mapping.options), human_votes=votes,
                          reference=(row.get(mapping.reference_column) or None) if mapping.reference_column else None,
                          prompt=prompt))
        if len(errors) >= 20:
            errors.append("... (more problems not shown)")
            break
    if errors:
        raise DatasetError("; ".join(errors))
    return items[:limit] if limit else items


def items_from_prompts(prompts: list[str], task_type: str = OPEN, options: tuple[str, ...] = ()) -> list[Item]:
    """Typed prompts (one per entry) -> items, for users who have no dataset file."""
    clean = [p.strip() for p in prompts if p and p.strip()]
    if not clean:
        raise DatasetError("enter at least one prompt")
    if len(clean) > MAX_ROWS:
        raise DatasetError(f"too many prompts ({len(clean)}); the limit is {MAX_ROWS}")
    if task_type == CLASSIFICATION and len(options) < 2:
        raise DatasetError("classification needs at least 2 options")
    return [Item(item_id=f"p{i}", task_type=task_type, instruction="", text=p, options=tuple(options), prompt=p)
            for i, p in enumerate(clean, start=1)]
