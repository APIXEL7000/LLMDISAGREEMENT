"""Plain-English explanation of a comparison result.

Everything here is computed from the run's actual numbers.  Fixed text is limited to definitions
("what this number means") and clearly-labelled illustrations; reasons are phrased as
*possible* explanations and always carry the evidence found in the data.  Nothing claims a cause
was proven.
"""
from __future__ import annotations

# plain names for the technical categories (technical names stay in exports and tooltips)
PLAIN = {
    "consensus": ("All agree", "Every AI gave the same answer."),
    "wording_only": ("Same answer, different words", "The AIs agree; only the wording differs."),
    "paraphrase_confirmed": ("Same meaning (checked by referee AI)", "A referee AI confirmed the differently worded answers mean the same."),
    "unverified_difference": ("Different words - can't tell yet", "The answers are worded differently and we could not confirm whether they mean the same. A referee AI can settle this."),
    "refusal_mismatch": ("One AI refused", "At least one AI declined or was blocked while another answered."),
    "majority_outlier": ("Most agree, one differs", "Most AIs agree but one (or more) gave a different answer."),
    "split_decision": ("No majority", "No answer was given by more than half of the AIs."),
    "not_comparable": ("Not enough answers", "Fewer than two AIs produced a usable answer."),
}
LEVELS = [(0.10, "low"), (0.30, "moderate"), (1.01, "high")]
WHY_IT_MATTERS = [
    "If two AIs give different answers to the same question, at least one of them is wrong, or the question is genuinely ambiguous. Either way a person should look at it.",
    "Agreement is a useful warning light: questions where the AIs disagree are the ones to double-check before relying on any single AI.",
    "Agreement does not prove an answer is correct - the AIs could all be wrong in the same way. This tool measures consistency, not truth.",
]
HOW_TO_READ = {
    "moderate": "Moderate disagreement: the AIs agree most of the time but differ often enough that the questions they differ on deserve a closer look.",
    "high": "High disagreement: the AIs often give different answers, so answers from a single AI should be checked, or the questions may be ambiguous.",
    "low": "Low disagreement: the AIs mostly agree. That is reassuring about consistency, but not proof that they are right.",
}


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _short(s: str, n: int = 70) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "..."


def level_for(rate: float | None) -> str | None:
    if rate is None:
        return None
    return next(label for cut, label in LEVELS if rate < cut)


def _example(c: dict, category: str) -> str | None:
    """A real example (from this run) for a category, or None."""
    ids = c["examples"].get(category) or []
    if not ids:
        return None
    r = next(x for x in c["items"] if x["item_id"] == ids[0])
    answers = "; ".join(f"{m} said \"{_short(a['text'] or a['error'], 40)}\"" for m, a in list(r["answers"].items())[:3])
    return f"Question {r['item_id']} (\"{_short(r['prompt'], 60)}\"): {answers}."


def explain_results(c: dict, summary: dict) -> dict:
    o, models, pairs = c["overall"], c["models"], [p for p in c["pairs"] if p["n_items"]]
    n, nt = o["n_compared"], o["n_items"]
    out: dict = {"category_labels": {k: {"label": v[0], "meaning": v[1]} for k, v in PLAIN.items()},
                 "why_it_matters": WHY_IT_MATTERS, "how_to_read": HOW_TO_READ, "simulated": bool(c.get("simulated"))}
    if n == 0:
        out.update(headline="No question had usable answers from at least two AIs, so disagreement could not be measured.",
                   level=None, metrics=[], reasons=[], model_notes=[], technical=[])
        return out

    lvl = level_for(o["disagreement_rate"])
    caveat = " (this is a rule of thumb used by this dashboard, not a scientific standard)"
    lo, hi = o["disagreement_ci95"]
    out["headline"] = (f"{n} of {nt} questions could be compared. The AIs clearly disagreed on {o['disagree_n']} ({_pct(o['disagreement_rate'])}) "
                       f"and agreed on {o['agree_n']} ({_pct(o['agreement_rate'])})"
                       + (f"; {o['unverified_n']} could not be judged yet." if o["unverified_n"] else "."))
    out["level"] = {"label": lvl, "text": f"Overall disagreement is {lvl.upper()}{caveat}. " + HOW_TO_READ[lvl]
                    + (f" Only {n} questions were compared, so treat this as a first indication." if n < 30 else "")}

    # ---- the number cards ----
    ex = lambda cat, fallback: _example(c, cat) or fallback
    metrics = [
        {"key": "agreement", "name": "Agreement", "value": _pct(o["agreement_rate"]),
         "meaning": "How often the AIs gave the same answer, even if they worded it differently.",
         "reading": "Higher = more consistent AIs.",
         "example": ex("consensus", "Illustration: three AIs all answer \"Canberra\" -> agreement.")},
        {"key": "disagreement", "name": "Clear disagreement", "value": _pct(o["disagreement_rate"]),
         "meaning": f"Questions where at least two AIs gave answers with a different meaning. Because only {n} questions were tested, "
                    f"the true rate could plausibly be anywhere from {_pct(lo)} to {_pct(hi)}.",
         "reading": "Higher = the AIs contradict each other more often.",
         "example": ex("majority_outlier", ex("split_decision", ex("refusal_mismatch", "No clear disagreement was found in this run.")))},
        {"key": "wording", "name": "Same answer, different words", "value": _pct(o["wording_only_rate"]),
         "meaning": "Questions where the AIs agree but phrase it differently. These are NOT counted as disagreement.",
         "reading": "Shows why comparing raw text would exaggerate disagreement.",
         "example": ex("wording_only", ex("paraphrase_confirmed", "Illustration: \"Paris\" vs \"The capital is Paris\" -> same answer."))},
        {"key": "unverified", "name": "Can't tell yet", "value": _pct(o["unverified_rate"]),
         "meaning": "Different wording that rules alone cannot classify as the same or different meaning. Adding a referee AI resolves these.",
         "reading": (f"If all were really the same idea, disagreement stays {_pct(o['disagreement_rate'])}; if all were really different, "
                     f"it could reach {_pct(o['disagreement_upper_bound'])}."),
         "example": ex("unverified_difference", "None in this run.")},
        {"key": "coverage", "name": "Questions compared", "value": f"{n} / {nt}",
         "meaning": "Questions where at least two AIs produced a usable answer.",
         "reading": "More questions = more trustworthy percentages.", "example": ""},
    ]
    if pairs:
        top = max(pairs, key=lambda p: p["disagree_pct"])
        low = min(pairs, key=lambda p: p["disagree_pct"])
        metrics.append({"key": "pairs", "name": "Most different pair of AIs", "value": _pct(top["disagree_pct"]),
                        "meaning": f"{top['a']} vs {top['b']}: they clearly differed on {top['disagree']} of the {top['n_items']} questions both answered. "
                                   "Each pair of AIs is measured the same way as the overall disagreement.",
                        "reading": f"Lowest pair: {low['a']} vs {low['b']} ({_pct(low['disagree_pct'])}).",
                        "example": ""})
    odd = {m: s["sole_dissenter_items"] for m, s in models.items() if s["sole_dissenter_items"]}
    metrics.append({"key": "odd", "name": "Odd one out", "value": ", ".join(f"{m}: {k}" for m, k in odd.items()) or "none",
                    "meaning": "How many times an AI was the one disagreeing with the others. It describes disagreement, not quality.",
                    "reading": "A high count means this AI often differs from the group - it is not automatically the wrong one.", "example": ""})
    refusals = sum(s["refusals"] for s in models.values())
    metrics.append({"key": "refusals", "name": "Refusals", "value": str(refusals),
                    "meaning": "Answers where an AI declined (or the provider blocked the request).",
                    "reading": "Different AIs follow different safety rules.", "example": ex("refusal_mismatch", "")})
    out["metrics"] = metrics

    # ---- differences between models (only what the data shows) ----
    notes = []
    if pairs and len(pairs) > 1:
        top, low = max(pairs, key=lambda p: p["disagree_pct"]), min(pairs, key=lambda p: p["disagree_pct"])
        overlap = top["disagree_ci95"][0] <= low["disagree_ci95"][1]
        notes.append(f"Most different pair: {top['a']} and {top['b']} ({_pct(top['disagree_pct'])}). Most similar: {low['a']} and {low['b']} ({_pct(low['disagree_pct'])})."
                     + (" The ranges overlap, so this difference could be chance with this many questions." if overlap else ""))
    for m, k in odd.items():
        notes.append(f"{m} was the odd one out on {k} question{'s' if k != 1 else ''}.")
    words = {m: s["mean_words"] for m, s in models.items() if s["mean_words"]}
    if len(words) > 1 and max(words.values()) >= 1.5 * min(words.values()):
        hi_m, lo_m = max(words, key=words.get), min(words, key=words.get)
        notes.append(f"Answer length differs: {hi_m} averaged {words[hi_m]} words, {lo_m} averaged {words[lo_m]}.")
    bad = {m: s["errors"] for m, s in models.items() if s["errors"]}
    if bad:
        notes.append("Failed requests: " + ", ".join(f"{m} ({k})" for m, k in bad.items()) + " - excluded for those questions.")
    out["model_notes"] = notes

    # ---- possible reasons, each tied to evidence in this run ----
    cnt = o["category_counts"]
    reasons = []
    ww = cnt["wording_only"] + cnt["paraphrase_confirmed"]
    if ww:
        reasons.append({"reason": "Different writing style", "evidence": f"{ww} question(s) had the same answer in different words."})
    if cnt["unverified_difference"]:
        reasons.append({"reason": "Same idea phrased very differently, or genuinely different answers",
                        "evidence": f"{cnt['unverified_difference']} question(s) could not be classified without a referee AI."})
    if cnt["refusal_mismatch"]:
        who = sorted({m for r in c["items"] if r["category"] == "refusal_mismatch" for m in r["dissenting_models"]})
        reasons.append({"reason": "Different safety rules", "evidence": f"{cnt['refusal_mismatch']} question(s) where one AI refused while another answered ({', '.join(who)})."})
    real = cnt["majority_outlier"] + cnt["split_decision"]
    if real:
        reasons.append({"reason": "Real differences in knowledge or reasoning, or an ambiguous question",
                        "evidence": f"{real} question(s) with different answers (see the examples). Read them to judge which."})
    if len(words) > 1 and max(words.values()) >= 2 * min(words.values()):
        reasons.append({"reason": "Different answer length", "evidence": "Average length differs by 2x or more between AIs; longer answers have more room to differ."})
    out["reasons"] = reasons
    out["reasons_note"] = "These are possible explanations that fit the data. This tool does not prove why the AIs differ."

    # ---- technical metrics kept, each with a plain meaning ----
    tech = [
        {"name": "Mean agreement ratio", "value": summary.get("mean_agreement"), "meaning": "Average share of identical answers per question, counting every response (all samples) together."},
        {"name": "Mean normalised entropy", "value": summary.get("mean_entropy_norm"), "meaning": "Average 'spread' of answers: 0 = always the same answer, 1 = as spread out as possible."},
        {"name": "Questions flagged (agreement below threshold)", "value": f"{summary.get('n_flagged')} ({summary.get('pct_flagged')}%)", "meaning": f"Questions where the share of identical answers fell below {summary.get('agreement_threshold')}."},
        {"name": "Cross-model disagreements", "value": summary.get("n_cross_model_disagree"), "meaning": "Questions where different models' most common answers differ."},
        {"name": "Confidence interval (95%)", "value": f"{_pct(lo)} - {_pct(hi)}", "meaning": "Range in which the true disagreement rate plausibly lies (Wilson score interval)."},
    ]
    hc = summary.get("human_comparison")
    if hc:
        tech += [{"name": "Mean JS divergence vs humans", "value": hc["mean_js_divergence"], "meaning": "How far the AI answer distribution is from human votes (0 = identical, 1 = completely different)."},
                 {"name": "AI majority matches human majority", "value": _pct(hc["majority_match_rate"]), "meaning": "Share of questions where the most common AI answer equals the most common human answer."},
                 {"name": "Overconfident questions", "value": hc["n_overconfident"], "meaning": "Humans were split but the AI answered the same way every time."}]
    out["technical"] = tech
    return out
