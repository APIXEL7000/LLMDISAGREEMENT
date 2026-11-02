"""Downloadable report: ONE self-contained HTML file (no external resources).

Open it in any browser; use Print -> "Save as PDF" for a PDF.  Contains only results that were
computed in the run, the method and its limits, and the run's state trace.  No API keys, ever.
"""
from __future__ import annotations

from datetime import datetime
from html import escape as e

from .comparison import CATEGORY_TEXT


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _ci(ci) -> str:
    return "" if not ci or ci[0] != ci[0] else f" ({_pct(ci[0])}-{_pct(ci[1])})"


CSS = """
body{font:14px/1.55 system-ui,Segoe UI,Helvetica,Arial,sans-serif;color:#1d2433;max-width:980px;margin:24px auto;padding:0 18px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:26px 0 8px;border-bottom:2px solid #dfe3ec;padding-bottom:4px}h3{font-size:14px;margin:16px 0 6px}
table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0}th,td{border:1px solid #dfe3ec;padding:5px 8px;text-align:left;vertical-align:top}th{background:#f3f5fa}
.meta{color:#667085;font-size:12.5px}.warn{background:#fbe0df;border:1px solid #d9534f;padding:9px 12px;border-radius:6px;font-weight:600;margin:12px 0}
.kpis{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}.kpi{border:1px solid #dfe3ec;border-radius:8px;padding:8px 12px;min-width:150px}.kpi b{display:block;font-size:22px}.kpi span{font-size:11.5px;color:#667085}
.resp{border:1px solid #dfe3ec;border-radius:6px;padding:5px 9px;margin:4px 0;background:#fafbfd}.resp.d{border-color:#d9534f}.resp pre{margin:2px 0 0;white-space:pre-wrap;font:12px/1.4 inherit}
.pill{display:inline-block;padding:0 8px;border-radius:10px;font-size:11px;color:#fff;background:#667085}.agree{background:#1f9d63}.disagreement{background:#d9534f}.unverified{background:#d9a21b}
.ex{break-inside:avoid;margin:8px 0}ul li{margin:4px 0}svg{max-width:100%;height:auto}
@media print{body{margin:0;max-width:none}h2{break-after:avoid}table,.kpi,.resp{break-inside:avoid}}
"""
GROUP = {"consensus": "agree", "wording_only": "agree", "paraphrase_confirmed": "agree", "unverified_difference": "unverified",
         "refusal_mismatch": "disagreement", "majority_outlier": "disagreement", "split_decision": "disagreement",
         "not_comparable": ""}
METHOD = [
    "Each selected model received exactly the same prompt for every input; the responses shown are the providers' actual replies.",
    "Answers were normalised (labels extracted for classification; lower-casing and punctuation removal for free text). Refusals and provider safety blocks are kept as their own kind of answer.",
    "Two answers were judged <b>same</b> by rules when they are identical after normalisation, or share almost all content words (or one contains the other with at most two extra words) and contain the same numbers and the same negation words. Different numbers, one-sided negation, different classification labels, or a refusal versus an answer are judged <b>different</b>.",
    "Free-text pairs the rules cannot settle were sent to the judge model (SAME / DIFFERENT) when one was selected; otherwise they are labelled <b>unverified</b> and are counted neither as agreement nor as disagreement. The confirmed rate and the upper bound are both reported.",
    "Rates are computed over inputs with at least two usable answers. Intervals are 95% Wilson score intervals. Pairwise rates use the inputs both models answered.",
]
LIMITS = [
    "Agreement between models does not mean the answer is correct; this report does not measure correctness.",
    "The judge model, when used, is itself an LLM and can be wrong; refusal detection uses wording heuristics and API block signals.",
    "A model that failed on an input is excluded for that input. Small samples give wide intervals; differences between pairs may be chance.",
]


def build_html_report(comparison: dict, summary: dict, config: dict, svg: str, trace: list[dict], run_id: str,
                      generated: datetime | None = None) -> str:
    c, o = comparison, comparison["overall"]
    generated = generated or datetime.now()
    p = []
    add = p.append
    add(f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>Disagreement report {e(run_id)}</title><style>{CSS}</style></head><body>")
    add("<h1>LLM Disagreement Analysis Report</h1>")
    add(f"<div class='meta'>Run {e(run_id)} &middot; generated {generated:%Y-%m-%d %H:%M} &middot; "
        f"{o['n_items']} inputs &middot; models: {e(', '.join(config['models']))}<br>"
        f"responses per model per input: {config['samples']} &middot; temperature {config['temperature']} &middot; "
        f"judge model: {e(config['judge'] or 'none (rule-based checks only)')}</div>")
    if comparison.get("simulated"):
        add("<div class='warn'>SIMULATED RUN: at least one model is the mock simulator, so these numbers test the pipeline only "
            "and say nothing about real language models.</div>")
    ex = c.get("explain")
    if ex and ex.get("metrics"):
        add(f"<h2>Summary in plain English</h2><p><b>{e(ex['headline'])}</b></p><p>{e(ex['level']['text'])}</p>")
        if ex["model_notes"]:
            add("<ul>" + "".join(f"<li>{e(n)}</li>" for n in ex["model_notes"]) + "</ul>")
        if ex["reasons"]:
            add("<h3>Possible reasons for the disagreements</h3><ul>" + "".join(
                f"<li><b>{e(r['reason'])}</b> - {e(r['evidence'])}</li>" for r in ex["reasons"]) + f"</ul><p class='meta'>{e(ex['reasons_note'])}</p>")
        add("<h3>What each number means</h3><table><tr><th>Number</th><th>Value</th><th>Meaning</th><th>Example from this run</th></tr>" + "".join(
            f"<tr><td>{e(m['name'])}</td><td>{e(m['value'])}</td><td>{e(m['meaning'])} {e(m['reading'])}</td><td>{e(m['example'])}</td></tr>"
            for m in ex["metrics"]) + "</table><h3>Why disagreement matters</h3><ul>" + "".join(f"<li>{e(w)}</li>" for w in ex["why_it_matters"]) + "</ul>")
    add("<h2>Key findings</h2><ul>" + "".join(f"<li>{e(f)}</li>" for f in c["findings"]) + "</ul>")
    add("<h2>Overview</h2><div class='kpis'>"
        f"<div class='kpi'><b>{o['n_compared']}/{o['n_items']}</b><span>inputs compared (&ge;2 models answered)</span></div>"
        f"<div class='kpi'><b>{_pct(o['agreement_rate'])}</b><span>models agree ({_pct(o['wording_only_rate'])} despite different wording)</span></div>"
        f"<div class='kpi'><b>{_pct(o['disagreement_rate'])}</b><span>confirmed disagreement{e(_ci(o['disagreement_ci95']))}</span></div>"
        f"<div class='kpi'><b>{_pct(o['unverified_rate'])}</b><span>unverified differences (upper bound on disagreement: {_pct(o['disagreement_upper_bound'])})</span></div></div>")
    add("<h2>Categories</h2><table><tr><th>Category</th><th>Meaning</th><th>Inputs</th><th>Share of all inputs</th></tr>")
    for k, n in o["category_counts"].items():
        if n:
            add(f"<tr><td><span class='pill {GROUP.get(k, '')}'>{e(k.replace('_', ' '))}</span></td><td>{e(CATEGORY_TEXT[k])}</td>"
                f"<td>{n}</td><td>{_pct(n / o['n_items'])}</td></tr>")
    add("</table>")
    pairs = [x for x in c["pairs"] if x["n_items"]]
    add("<h2>Pairwise disagreement</h2>")
    if pairs:
        add("<table><tr><th>Model A</th><th>Model B</th><th>Inputs</th><th>Identical</th><th>Same meaning, different wording</th>"
            "<th>Unverified</th><th>Disagree</th><th>Disagreement rate (95% interval)</th></tr>")
        for x in pairs:
            add(f"<tr><td>{e(x['a'])}</td><td>{e(x['b'])}</td><td>{x['n_items']}</td><td>{x['identical']}</td><td>{x['wording']}</td>"
                f"<td>{x['unverified']}</td><td>{x['disagree']}</td><td>{_pct(x['disagree_pct'])}{e(_ci(x['disagree_ci95']))}</td></tr>")
        add("</table>")
    else:
        add("<p>Pairwise rates need at least two models with answers.</p>")
    add("<h2>Models</h2><table><tr><th>Model</th><th>Requests</th><th>OK</th><th>Refusals/blocked</th><th>Errors</th><th>Avg words</th>"
        "<th>Sole dissenter on</th><th>Status</th></tr>")
    rep = c.get("model_report") or {}
    for m, x in c["models"].items():
        why = (rep.get(m) or {}).get("disabled_reason")
        add(f"<tr><td>{e(m)}{' (simulated)' if x['simulated'] else ''}</td><td>{x['requests']}</td><td>{x['ok']}</td><td>{x['refusals']}</td>"
            f"<td>{x['errors']}</td><td>{x['mean_words'] if x['mean_words'] is not None else '-'}</td><td>{x['sole_dissenter_items']}</td>"
            f"<td>{'disabled: ' + e(why[:160]) if why else 'ok'}</td></tr>")
    for m, r in rep.items():
        if m not in c["models"]:
            add(f"<tr><td>{e(m)}</td><td colspan='6'>no answers</td><td>{e((r.get('disabled_reason') or '')[:160])}</td></tr>")
    add("</table><p class='meta'>\"Sole dissenter\" counts inputs where this model was outside the majority; it describes disagreement, not quality.</p>")

    hc = summary.get("human_comparison")
    if hc:
        add(f"<h2>Comparison with human labels</h2><p>{hc['n_items_with_human_votes']} inputs had human votes. Mean Jensen-Shannon divergence "
            f"{hc['mean_js_divergence']} (0 = identical distributions); LLM majority equals human majority on {_pct(hc['majority_match_rate'])}; "
            f"{hc['n_overconfident']} overconfident inputs (LLM consistent while humans split).</p>")

    def item_html(r):
        out = [f"<div class='ex'><b>{e(r['item_id'])}</b> <span class='pill {GROUP.get(r['category'], '')}'>{e(r['category'].replace('_', ' '))}</span>"
               f"<pre style='white-space:pre-wrap;margin:3px 0'>{e(r['prompt'])}</pre>"]
        for m, a in r["answers"].items():
            body = a["error"] if a["status"] == "error" else a["text"]
            flag = " (" + a["status"] + ")" if a["status"] != "ok" else ""
            out.append(f"<div class='resp {'d' if m in r['dissenting_models'] else ''}'><b>{e(m)}</b>{e(flag)}<pre>{e(body) or '(empty)'}</pre></div>")
        return "".join(out) + "</div>"

    add("<h2>Examples by category</h2>")
    by_id = {r["item_id"]: r for r in c["items"]}
    shown = False
    for k, ids in c["examples"].items():
        if ids:
            shown = True
            add(f"<h3>{e(k.replace('_', ' '))} <span class='meta'>- {e(CATEGORY_TEXT[k])}</span></h3>" + "".join(item_html(by_id[i]) for i in ids))
    if not shown:
        add("<p>No examples.</p>")
    add("<h2>All inputs</h2><table><tr><th>Id</th><th>Prompt</th><th>Category</th><th>Dissenting models</th></tr>")
    for r in c["items"][:300]:
        add(f"<tr><td>{e(r['item_id'])}</td><td>{e(r['prompt'][:140])}</td><td>{e(r['category'].replace('_', ' '))}</td>"
            f"<td>{e(', '.join(r['dissenting_models']))}</td></tr>")
    add("</table>" + ("<p class='meta'>First 300 shown; the full per-input table is in items_comparison.csv.</p>" if len(c["items"]) > 300 else ""))
    add("<h2>How disagreement was measured</h2><ol>" + "".join(f"<li>{m}</li>" for m in METHOD) + "</ol>")
    add("<h2>Limitations</h2><ul>" + "".join(f"<li>{m}</li>" for m in LIMITS) + "</ul>")
    add("<h2>Pipeline state trace</h2>" + svg + "<table><tr><th>#</th><th>From</th><th>Event [guard]</th><th>To</th><th>Note</th></tr>")
    for t in trace:
        g = f" [{e(t['guard'])}]" if t["guard"] else ""
        add(f"<tr><td>{t['seq']}</td><td>{e(t['from'])}</td><td>{e(t['event'])}{g}</td><td>{e(t['to'])}</td><td>{e(t['note'])}</td></tr>")
    add("</table><p class='meta'>This report contains no API keys. Responses are reproduced as received from the providers.</p></body></html>")
    return "".join(p)
