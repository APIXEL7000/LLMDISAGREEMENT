"""Cross-model comparison  (STD states EVALUATING + DETECTING).

Question answered: for the SAME input, do the selected models give the same answer?

How "same" is decided (three tiers, strongest evidence first)
-------------------------------------------------------------
1. Rule-based, certain
   * classification items: the extracted labels are equal / different;
   * one model refused (or was safety-blocked) and another answered  -> different;
   * after removing stop-words the answers are equal, or overlap >= `wording_threshold`,
     or one contains the other with at most 2 extra words, AND they contain the same
     numbers and the same negation words                           -> same, wording differs.
   * different numbers or different negation words                 -> different.
2. Judge model (optional, `cfg.judge_model`): for free-text answers that the rules cannot
   settle, a model is asked "SAME or DIFFERENT?".  Its verdict is recorded as
   judge-confirmed, and it is itself an LLM, so it can be wrong.
3. Unresolved: without a judge, lexically different free-text answers are labelled
   `unverified` - they are NOT counted as disagreement and NOT as agreement.  Both the
   confirmed rate and the upper bound (confirmed + unverified) are reported.

Nothing here claims to measure correctness: two models can agree and both be wrong.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from itertools import combinations
from typing import Callable

import pandas as pd

from .config import Config
from .data_collection import CLASSIFICATION, Item, build_prompt
from .llm_clients import GenRequest, LLMClient, LLMError
from .normalization import REFUSAL

STOP = set("a an the is are was were be been being of to in on at for and or it its this that these those "
           "as by with from has have had do does did so than then there their they he she we you i".split())
NEG = {"not", "no", "never", "none", "neither", "nor", "without", "cannot", "cant", "dont", "doesnt",
       "didnt", "isnt", "arent", "wasnt", "werent", "wont", "wouldnt", "shouldnt", "couldnt", "hasnt",
       "havent", "hadnt", "nothing", "nobody"}
AGREE_CATS = ("consensus", "wording_only", "paraphrase_confirmed")
DISAGREE_CATS = ("refusal_mismatch", "majority_outlier", "split_decision")
CATEGORY_TEXT = {
    "consensus": "All models gave the same answer.",
    "wording_only": "Models agree; the wording differs but the content is the same (rule-based).",
    "paraphrase_confirmed": "Models agree; a judge model confirmed the differently worded answers mean the same.",
    "unverified_difference": "Answers are worded differently and could not be verified as the same or different (no judge model).",
    "refusal_mismatch": "At least one model refused or was blocked while another answered.",
    "majority_outlier": "Most models agree; one or more models gave a different answer.",
    "split_decision": "No answer was given by a majority of the models.",
    "not_comparable": "Fewer than two models produced a usable answer.",
}


# ----------------------------------------------------------------------------
# Tier 1: rule-based verdicts
# ----------------------------------------------------------------------------
def content_tokens(answer: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", answer.lower().replace("'", "").replace("\u2019", ""))
    return [t for t in toks if t not in STOP]


def lexical_verdict(a: str, b: str, task_type: str, threshold: float) -> str:
    """'different' | 'wording' | 'unverified' for two normalised, non-identical answers."""
    if a == b:
        return "wording"
    if REFUSAL in (a, b) or task_type == CLASSIFICATION:
        return "different"
    ta, tb = content_tokens(a), content_tokens(b)
    sa, sb = set(ta), set(tb)
    if not sa or not sb:
        return "unverified"
    if {t for t in sa if t[0].isdigit()} != {t for t in sb if t[0].isdigit()}:
        return "different"                                   # different numbers
    if (sa & NEG) != (sb & NEG):
        return "different"                                   # one negates, the other does not
    jac = len(sa & sb) / len(sa | sb)
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    contained = set(small) <= set(big) and len(big) <= len(small) + 2
    return "wording" if (jac >= threshold or contained) else "unverified"


# ----------------------------------------------------------------------------
# Tier 2: judge model
# ----------------------------------------------------------------------------
JUDGE_PROMPT = ("You are checking whether two answers to the same question agree.\n\nQuestion:\n{q}\n\n"
                "Answer A:\n{a}\n\nAnswer B:\n{b}\n\n"
                "Do A and B give the same answer or conclusion to the question? Ignore differences in wording, "
                "length, formatting and level of detail. Treat different facts, numbers or opposite conclusions "
                "as different. Reply with exactly one word: SAME or DIFFERENT.")


def parse_judge(text: str) -> bool | None:
    up = text.upper()
    if re.search(r"\bDIFFERENT\b", up):
        return False
    if re.search(r"\bSAME\b", up):
        return True
    return None


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def assign_clusters(items: list[Item], responses: pd.DataFrame, cfg: Config,
                    judge: LLMClient | None = None,
                    progress: Callable[[int, int, str], None] | None = None) -> tuple[pd.DataFrame, dict]:
    """Add `cluster` (lenient) and `cluster_strict` columns; return (responses, stage report).

    strict  : merges only certain/judged equivalences.
    lenient : additionally merges `unverified` pairs (so remaining differences are certain).
    """
    out = responses.copy()
    out["cluster"], out["cluster_strict"] = -1, -1
    pair_log: list[dict] = []
    rep = {"judge_used": judge is not None, "judge_calls": 0, "judge_errors": 0,
           "judge_disabled": None, "pair_log": pair_log}
    cache: dict[tuple, bool | None] = {}
    streak = 0
    valid = out[out["valid"]] if len(out) else out
    by_item = {k: g for k, g in valid.groupby("item_id")}
    pending = 0
    if judge:        # count first, for the progress bar
        for it in items:
            g = by_item.get(it.item_id)
            if g is not None and it.task_type != CLASSIFICATION:
                u = g["answer"].unique()
                pending += sum(lexical_verdict(a, b, it.task_type, cfg.wording_threshold) == "unverified"
                               for a, b in combinations(u, 2))
    done = 0
    for it in items:
        g = by_item.get(it.item_id)
        if g is None or g.empty:
            continue
        counts = Counter(g["answer"])
        uniq = sorted(counts, key=lambda a: (-counts[a], a))
        raw = g.drop_duplicates("answer").set_index("answer")["raw_text"].to_dict()
        n = len(uniq)
        strict, lenient = _UnionFind(n), _UnionFind(n)
        for i, j in combinations(range(n), 2):
            a, b = uniq[i], uniq[j]
            v = lexical_verdict(a, b, it.task_type, cfg.wording_threshold)
            method = "rule"
            if v == "unverified" and judge and not rep["judge_disabled"]:
                key = (it.item_id, a, b)
                if key not in cache:
                    cache[key] = _ask_judge(judge, it, raw[a], raw[b], cfg, rep)
                    streak = streak + 1 if cache[key] is None else 0
                    if streak >= 3:
                        rep["judge_disabled"] = "3 consecutive judge failures; remaining pairs left unverified"
                    done += 1
                    if progress:
                        progress(done, max(pending, 1), f"judging answers - item {it.item_id}")
                verdict = cache[key]
                if verdict is True:
                    v, method = "equivalent", "judge"
                elif verdict is False:
                    v, method = "different", "judge"
            if v in ("wording", "equivalent"):
                strict.union(i, j); lenient.union(i, j)
            elif v == "unverified":
                lenient.union(i, j)
            pair_log.append({"item_id": it.item_id, "answer_a": a[:200], "answer_b": b[:200],
                             "verdict": v, "method": method})
        cl = {a: lenient.find(k) for k, a in enumerate(uniq)}
        cs = {a: strict.find(k) for k, a in enumerate(uniq)}
        idx = out.index[(out["item_id"] == it.item_id) & out["valid"]]
        out.loc[idx, "cluster"] = out.loc[idx, "answer"].map(cl)
        out.loc[idx, "cluster_strict"] = out.loc[idx, "answer"].map(cs)
    return out, rep


def _ask_judge(judge: LLMClient, item: Item, a: str, b: str, cfg: Config, rep: dict) -> bool | None:
    prompt = JUDGE_PROMPT.format(q=build_prompt(item)[:2000], a=a[:1500], b=b[:1500])
    rep["judge_calls"] += 1
    try:
        reply = judge.generate(GenRequest(item.item_id, prompt, 0, 0.0, 20))
    except LLMError:
        rep["judge_errors"] += 1
        return None
    v = parse_judge(reply.text)
    if v is None:
        rep["judge_errors"] += 1
    return v


# ----------------------------------------------------------------------------
# Tier 3 + aggregation
# ----------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (as fractions)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _model_majority(g: pd.DataFrame, col: str):
    """(cluster, representative answer) of the model's most frequent cluster for one item."""
    cc = Counter(g[col])
    top = sorted(cc, key=lambda c: (-cc[c], c))[0]
    ans = Counter(g.loc[g[col] == top, "answer"])
    return top, sorted(ans, key=lambda a: (-ans[a], a))[0]

 
def _trim(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + "..."


def compare_models(items: list[Item], responses: pd.DataFrame, cfg: Config, stage: dict | None = None) -> dict:
    stage = stage or {}
    valid = responses[responses["valid"]] if len(responses) else responses
    all_models = list(dict.fromkeys(responses["model"])) if len(responses) else []
    # only models that produced at least one usable answer take part in comparisons and rates
    models = [m for m in all_models if (len(valid) and (valid["model"] == m).any())]
    by_item = {k: g for k, g in valid.groupby("item_id")}
    rows, pair_acc = [], {pair: Counter() for pair in combinations(models, 2)}
    outlier_count: Counter = Counter()
    judge_pairs = {(p["item_id"]) for p in stage.get("pair_log", []) if p["method"] == "judge" and p["verdict"] == "equivalent"}

    for it in items:
        g = by_item.get(it.item_id)
        per_model = {}
        if g is not None:
            for m, gm in g.groupby("model"):
                c_len, ans = _model_majority(gm, "cluster")
                c_str, _ = _model_majority(gm, "cluster_strict")
                per_model[m] = {"lenient": c_len, "strict": c_str, "answer": ans}
        # per-model raw text for display
        shown = {}
        for m in models:
            gm = responses[(responses["item_id"] == it.item_id) & (responses["model"] == m)]
            if gm.empty:
                continue
            first = gm.iloc[0]
            shown[m] = {"text": _trim(first["raw_text"], 600), "answer": first["answer"],
                        "status": first["status"], "error": _trim(first.get("error", ""), 200),
                        "n_samples": int(len(gm))}
        resp = list(per_model)
        cat, dissent = "not_comparable", []
        if len(resp) >= 2:
            L = {per_model[m]["lenient"] for m in resp}
            S = {per_model[m]["strict"] for m in resp}
            answers = {per_model[m]["answer"] for m in resp}
            if len(L) == 1:
                if len(S) > 1:
                    cat = "unverified_difference"
                elif len(answers) == 1:
                    cat = "consensus"
                else:
                    cat = "paraphrase_confirmed" if it.item_id in judge_pairs else "wording_only"
            else:
                sizes = Counter(per_model[m]["lenient"] for m in resp)
                big, big_n = sizes.most_common(1)[0]
                ref_clusters = {per_model[m]["lenient"] for m in resp if per_model[m]["answer"] == REFUSAL}
                non_ref = L - ref_clusters
                if ref_clusters and len(non_ref) <= 1:
                    cat = "refusal_mismatch"
                    dissent = [m for m in resp if per_model[m]["lenient"] in ref_clusters]
                elif big_n >= 2 and big_n > len(resp) / 2:
                    cat = "majority_outlier"
                    dissent = [m for m in resp if per_model[m]["lenient"] != big]
                else:
                    cat = "split_decision"
            for m in dissent:
                if cat == "majority_outlier":
                    outlier_count[m] += 1
            for a, b in pair_acc:
                if a in per_model and b in per_model:
                    pa, pb = per_model[a], per_model[b]
                    pc = pair_acc[(a, b)]
                    pc["n"] += 1
                    if pa["lenient"] != pb["lenient"]:
                        pc["disagree"] += 1
                    elif pa["strict"] != pb["strict"]:
                        pc["unverified"] += 1
                    elif pa["answer"] != pb["answer"]:
                        pc["wording"] += 1
                    else:
                        pc["identical"] += 1
        group = ("agreement" if cat in AGREE_CATS else "disagreement" if cat in DISAGREE_CATS
                 else "unverified" if cat == "unverified_difference" else "excluded")
        rows.append({"item_id": it.item_id, "prompt": _trim(build_prompt(it), 500), "category": cat,
                     "group": group, "n_models_answered": len(resp), "dissenting_models": dissent,
                     "answers": shown, "reference": it.reference})

    n_cmp = sum(r["group"] != "excluded" for r in rows)
    cnt = Counter(r["category"] for r in rows)
    agree = sum(cnt[c] for c in AGREE_CATS)
    dis = sum(cnt[c] for c in DISAGREE_CATS)
    unv = cnt["unverified_difference"]
    lo, hi = wilson(dis, n_cmp)
    overall = {
        "n_items": len(items), "n_compared": n_cmp, "n_models": len(models),
        "category_counts": {c: cnt.get(c, 0) for c in CATEGORY_TEXT},
        "agree_n": agree, "disagree_n": dis, "unverified_n": unv,
        "agreement_rate": agree / n_cmp if n_cmp else None,
        "disagreement_rate": dis / n_cmp if n_cmp else None,
        "disagreement_ci95": [lo, hi] if n_cmp else None,
        "unverified_rate": unv / n_cmp if n_cmp else None,
        "disagreement_upper_bound": (dis + unv) / n_cmp if n_cmp else None,
        "wording_only_rate": (cnt["wording_only"] + cnt["paraphrase_confirmed"]) / n_cmp if n_cmp else None,
    }
    pairs = []
    for (a, b), c in pair_acc.items():
        n = c["n"]
        plo, phi = wilson(c["disagree"], n)
        pairs.append({"a": a, "b": b, "n_items": n, "identical": c["identical"], "wording": c["wording"],
                      "unverified": c["unverified"], "disagree": c["disagree"],
                      "agree_pct": (c["identical"] + c["wording"]) / n if n else None,
                      "disagree_pct": c["disagree"] / n if n else None,
                      "unverified_pct": c["unverified"] / n if n else None,
                      "disagree_ci95": [plo, phi] if n else None})
    stats = {}
    for m in all_models:                                   # failed models still get a health row
        gm = responses[responses["model"] == m]
        ok = gm[gm["status"] == "ok"]
        stats[m] = {"requests": int(len(gm)), "ok": int((gm["status"] == "ok").sum()),
                    "blocked": int((gm["status"] == "blocked").sum()),
                    "errors": int((gm["status"] == "error").sum()),
                    "refusals": int((gm["answer"] == REFUSAL).sum()),
                    "mean_words": round(float(ok["raw_text"].str.split().str.len().mean()), 1) if len(ok) else None,
                    "sole_dissenter_items": outlier_count.get(m, 0),
                    "simulated": bool(gm["simulated"].any()) if "simulated" in gm and len(gm) else False}
    examples = {c: [r["item_id"] for r in rows if r["category"] == c][:3] for c in CATEGORY_TEXT}
    res = {"overall": overall, "pairs": pairs, "models": stats, "items": rows, "examples": examples,
           "category_text": CATEGORY_TEXT, "judge": {k: v for k, v in stage.items() if k != "pair_log"},
           "pair_log": stage.get("pair_log", [])}
    res["findings"] = findings(res)
    return res


# ----------------------------------------------------------------------------
# Plain-language findings: every sentence is computed from the numbers above
# ----------------------------------------------------------------------------
def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def findings(res: dict, simulated: bool = False) -> list[str]:
    o, out = res["overall"], []
    if o["n_compared"] == 0:
        return ["No item had usable answers from at least two models, so no disagreement could be measured."]
    lo, hi = o["disagreement_ci95"]
    out.append(f"{o['disagree_n']} of {o['n_compared']} comparable items ({_pct(o['disagreement_rate'])}) show confirmed "
               f"disagreement between models (95% interval {_pct(lo)}-{_pct(hi)}); {o['agree_n']} "
               f"({_pct(o['agreement_rate'])}) agree.")
    if o["wording_only_rate"]:
        out.append(f"{_pct(o['wording_only_rate'])} of comparable items agree despite different wording, so raw text "
                   "comparison alone would have overstated disagreement.")
    if o["unverified_n"]:
        n_u = o["unverified_n"]
        out.append(f"{n_u} item{'s' if n_u != 1 else ''} {'have' if n_u != 1 else 'has'} differently worded answers that could not be "
                   f"classified as the same or different. The true disagreement rate is therefore between {_pct(o['disagreement_rate'])} and "
                   f"{_pct(o['disagreement_upper_bound'])}. Select a judge model to resolve them.")
    pairs = [p for p in res["pairs"] if p["n_items"]]
    if len(pairs) >= 2:
        hi_p = max(pairs, key=lambda p: p["disagree_pct"])
        lo_p = min(pairs, key=lambda p: p["disagree_pct"])
        if hi_p is not lo_p:
            overlap = hi_p["disagree_ci95"][0] <= lo_p["disagree_ci95"][1]
            out.append(f"Highest pairwise disagreement: {hi_p['a']} vs {hi_p['b']} ({_pct(hi_p['disagree_pct'])}, "
                       f"n={hi_p['n_items']}); lowest: {lo_p['a']} vs {lo_p['b']} ({_pct(lo_p['disagree_pct'])}, "
                       f"n={lo_p['n_items']})."
                       + (" The intervals overlap, so this ranking may be chance at this sample size." if overlap else ""))
    cats = {c: n for c, n in o["category_counts"].items() if c in DISAGREE_CATS and n}
    if cats:
        top = max(cats, key=cats.get)
        out.append(f"Most common disagreement type: {top.replace('_', ' ')} ({cats[top]} item{'s' if cats[top] != 1 else ''}).")
    refusals = sum(m["refusals"] for m in res["models"].values())
    if refusals:
        out.append(f"{refusals} responses were refusals or provider blocks (detected by wording heuristics and API "
                   "block signals), which are treated as a distinct answer.")
    failed = sum(m["errors"] for m in res["models"].values())
    if failed:
        out.append(f"{failed} requests failed; the affected models are excluded for those items.")
    if o["n_compared"] < 30:
        out.append(f"Only {o['n_compared']} items were compared; treat percentages as indicative, not conclusive.")
    out.append("Agreement between models does not mean the answer is correct.")
    return out
