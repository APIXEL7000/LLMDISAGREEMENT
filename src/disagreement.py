"""Disagreement Detection Module  (DFD sub-process 3.3, STD state DETECTING).

Measures, per item, how much the collected responses disagree, and (when human
votes exist) how the LLM's answer distribution compares to the human one.

Metrics and the papers that motivate them (numbers = rows of the literature table)
-------------------------------------------------------------------------------
agreement            share of the majority answer           (25 self-consistency, 10/11)
entropy_bits/_norm   Shannon entropy of the answer dist.    (5, 8 consistency aggregation)
cluster entropy      entropy over meaning-clusters; here a token-overlap *proxy*
                     for semantic entropy                   (24 Farquhar et al.)
cross_model_*        do different models give different majorities?   (7 multi-model)
js_divergence, tv    LLM distribution vs human distribution (1, 2 Monte-Carlo estimate)
overconfident        LLM confident where humans disagree    (3 Lu et al.)
"""
from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from .config import Config
from .data_collection import CLASSIFICATION, Item


# ----------------------------------------------------------------------------
# Pure math helpers (unit-tested in tests/test_core.py)
# ----------------------------------------------------------------------------
def shannon_entropy(counts) -> float:
    """Shannon entropy in bits of a vector of counts."""
    c = np.asarray(list(counts), dtype=float)
    total = c.sum()
    if total <= 0:
        return 0.0
    p = c[c > 0] / total
    return float(-(p * np.log2(p)).sum() + 0.0)       # +0.0 avoids '-0.0'


def normalized_entropy(counts, k: int) -> float:
    """Entropy divided by its maximum log2(k), so it lies in [0, 1]."""
    return shannon_entropy(counts) / math.log2(k) if k > 1 else 0.0


def js_divergence(p, q) -> float:
    """Jensen-Shannon divergence, base 2, in [0, 1]."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def total_variation(p, q) -> float:
    return 0.5 * float(np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def cluster_map(answers: list[str], task_type: str, threshold: float) -> dict[str, int]:
    """Map every distinct answer to a cluster id.

    classification: identical labels share a cluster.
    open:           greedy token-overlap (Jaccard) clustering, most frequent
                    answers first.  Proxy for bidirectional-entailment clustering
                    in Farquhar et al. (2024) - see README 'Remaining 50%'.
    """
    counts = Counter(answers)
    order = sorted(counts, key=lambda a: (-counts[a], a))      # deterministic
    if task_type == CLASSIFICATION:
        return {a: i for i, a in enumerate(order)}
    reps: list[set] = []
    mapping: dict[str, int] = {}
    for a in order:
        toks = set(a.split())
        for cid, rep in enumerate(reps):
            if jaccard(toks, rep) >= threshold:
                mapping[a] = cid
                break
        else:
            reps.append(toks)
            mapping[a] = len(reps) - 1
    return mapping


# ----------------------------------------------------------------------------
# Per-item analysis
# ----------------------------------------------------------------------------
def _majority(answers: list[str], cmap: dict[str, int]) -> tuple[str, float, Counter]:
    """(representative answer of the largest cluster, its share, cluster counts)."""
    cc = Counter(cmap[a] for a in answers)
    top_cid = sorted(cc, key=lambda c: (-cc[c], c))[0]
    rep = Counter(a for a in answers if cmap[a] == top_cid)
    rep_answer = sorted(rep, key=lambda a: (-rep[a], a))[0]
    return rep_answer, cc[top_cid] / len(answers), cc


def analyze_item(item: Item, g: pd.DataFrame, cfg: Config) -> dict:
    valid = g[g["valid"]] if len(g) else g
    n_total, n_valid = len(g), len(valid)
    row: dict = {"item_id": item.item_id, "task_type": item.task_type,
                 "n_responses": n_total, "n_valid": n_valid,
                 "invalid_rate": (1 - n_valid / n_total) if n_total else np.nan}
    if n_valid == 0:
        row.update(majority_answer=None, agreement=np.nan, disagreement=True,
                   flag_reason="no_valid_answers")
        return row

    answers = valid["answer"].tolist()
    if "cluster" in valid and (valid["cluster"] >= 0).all():      # semantic clusters from comparison.py
        cmap = dict(zip(valid["answer"], valid["cluster"]))
    else:                                                        # standalone use: token-overlap proxy
        cmap = cluster_map(answers, item.task_type, cfg.jaccard_threshold)
    majority, agreement, cc = _majority(answers, cmap)
    k = len(item.options) if item.task_type == CLASSIFICATION else n_valid

    row.update(
        majority_answer=majority,
        agreement=round(agreement, 4),
        n_distinct_answers=len(set(answers)),
        n_clusters=len(cc),
        entropy_bits=round(shannon_entropy(cc.values()), 4),
        entropy_norm=round(normalized_entropy(cc.values(), k), 4),
        disagreement=agreement < cfg.agreement_threshold,
        flag_reason="low_agreement" if agreement < cfg.agreement_threshold else "",
    )

    # ---- cross-model comparison (paper 7) ----
    model_major, model_agree = {}, {}
    for m, gm in valid.groupby("model"):
        ans_m = gm["answer"].tolist()
        rep, ag, _ = _majority(ans_m, cmap)
        model_major[m], model_agree[m] = rep, ag
    cids = {cmap[a] for a in model_major.values()}
    row["model_majorities"] = "; ".join(f"{m}={a}" for m, a in sorted(model_major.items()))
    row["mean_model_self_agreement"] = round(float(np.mean(list(model_agree.values()))), 4)
    row["cross_model_disagree"] = len(cids) > 1
    pairs = list(combinations(model_major.values(), 2))
    row["pairwise_model_agreement"] = (
        round(sum(cmap[a] == cmap[b] for a, b in pairs) / len(pairs), 4) if pairs else np.nan)

    # ---- comparison with the human distribution (papers 1, 2, 3) ----
    q = item.human_distribution()
    if q is not None and item.task_type == CLASSIFICATION:
        counts = Counter(valid["answer"])
        p = np.array([counts.get(o, 0) for o in item.options], float) / n_valid  # Monte-Carlo estimate
        human_major = item.options[int(np.argmax(q))]
        human_agree = float(q.max())
        row.update(
            human_majority=human_major,
            human_agreement=round(human_agree, 4),
            human_entropy_norm=round(normalized_entropy(q, len(item.options)), 4),
            js_divergence=round(js_divergence(p, q), 4),
            total_variation=round(total_variation(p, q), 4),
            majority_match=(majority == human_major),
            overconfident=(human_agree < cfg.human_agreement_threshold
                           and agreement >= cfg.llm_overconfidence_threshold),
        )
    return row


def analyze_all(items: list[Item], responses: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    groups = {k: v for k, v in responses.groupby("item_id")} if len(responses) else {}
    empty = responses.iloc[0:0]
    rows = [analyze_item(it, groups.get(it.item_id, empty), cfg) for it in items]
    df = pd.DataFrame(rows)
    for col in ("agreement", "entropy_norm", "invalid_rate", "cross_model_disagree"):
        if col not in df:                                        # e.g. no item had a valid answer
            df[col] = np.nan
    return df


# ----------------------------------------------------------------------------
# Dataset-level summary
# ----------------------------------------------------------------------------
def _pearson(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize(m: pd.DataFrame, cfg: Config) -> dict:
    s: dict = {
        "n_items": int(len(m)),
        "n_flagged": int(m["disagreement"].sum()),
        "pct_flagged": round(100 * float(m["disagreement"].mean()), 1),
        "mean_agreement": round(float(m["agreement"].mean()), 4),
        "mean_entropy_norm": round(float(m["entropy_norm"].mean()), 4),
        "mean_invalid_rate": round(float(m["invalid_rate"].mean()), 4),
        "n_cross_model_disagree": int(m["cross_model_disagree"].fillna(False).sum()),
        "agreement_threshold": cfg.agreement_threshold,
    }
    if "human_agreement" in m and m["human_agreement"].notna().any():
        h = m[m["human_agreement"].notna()]
        s["human_comparison"] = {
            "n_items_with_human_votes": int(len(h)),
            "mean_js_divergence": round(float(h["js_divergence"].mean()), 4),
            "mean_total_variation": round(float(h["total_variation"].mean()), 4),
            "majority_match_rate": round(float(h["majority_match"].astype(float).mean()), 4),
            "n_overconfident": int(h["overconfident"].sum()),
            "pearson_llm_vs_human_entropy": _pearson(h["entropy_norm"], h["human_entropy_norm"]),
        }
        hi = h[h["human_agreement"] >= cfg.human_agreement_threshold]
        lo = h[h["human_agreement"] < cfg.human_agreement_threshold]
        s["human_comparison"]["by_human_agreement"] = {
            "high_human_agreement": {"n": int(len(hi)),
                "mean_llm_agreement": _r(hi["agreement"]),
                "majority_match_rate": _r(hi["majority_match"].astype(float))},
            "low_human_agreement": {"n": int(len(lo)),
                "mean_llm_agreement": _r(lo["agreement"]),
                "majority_match_rate": _r(lo["majority_match"].astype(float))},
        }
    return s


def _r(series: pd.Series):
    return round(float(series.mean()), 4) if len(series) else None
