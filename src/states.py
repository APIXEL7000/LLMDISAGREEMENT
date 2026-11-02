"""State Transition Diagram (STD) - single source of truth.

Why this module exists
----------------------
Previously the states lived in four unsynchronised places (log strings in the
pipeline, a regex in the server, a hard-coded list in the web page, and a
hand-drawn .drawio file).  The run's path was *reconstructed by scraping log
text*, so any state that was not logged (INITIAL, FINAL), any transition (only
states were kept) and the whole trace (when INFO logging was off) could be
silently lost.

Now:
* `STATES` and `TRANSITIONS` define the diagram once.
* `StateMachine.fire()` is the only way the pipeline moves between states; it
  refuses transitions that are not in the table and records every move.
* `render_svg()` / `to_drawio()` draw ALL states and transitions from the table
  and overlay what the run actually did, so nothing valid can be omitted.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from html import escape


class State(str, Enum):
    INITIAL = "INITIAL"
    AWAITING_INPUT = "AWAITING_INPUT"
    CONFIGURING = "CONFIGURING"
    DATASET_READY = "DATASET_READY"
    QUERYING = "QUERYING"
    NORMALIZING = "NORMALIZING"
    EVALUATING = "EVALUATING"
    DETECTING = "DETECTING"
    RESOLVING = "RESOLVING"
    RESULTS_STORED = "RESULTS_STORED"
    GENERATING_REPORT = "GENERATING_REPORT"
    REPORT_DELIVERED = "REPORT_DELIVERED"
    FINAL = "FINAL"


@dataclass(frozen=True)
class StateInfo:
    label: str
    kind: str = "state"          # initial | state | final
    implemented: bool = True     # False = in the design (STD) but not coded yet
    activity: str = ""           # entry/do text shown as tooltip


STATES: dict[State, StateInfo] = {
    State.INITIAL: StateInfo("Initial", "initial"),
    State.AWAITING_INPUT: StateInfo("Awaiting Input", activity="wait for prompts, dataset and settings"),
    State.CONFIGURING: StateInfo("Configuring (P1)", activity="validate settings, map columns, build prompts"),
    State.DATASET_READY: StateInfo("Dataset Ready", activity="prompts prepared and stored in D1"),
    State.QUERYING: StateInfo("Querying LLMs (P2)", activity="send each prompt to every selected model"),
    State.NORMALIZING: StateInfo("Normalizing (3.1)", activity="clean responses, extract answers, detect refusals"),
    State.EVALUATING: StateInfo("Evaluating (3.2)", activity="semantic equivalence check of differing answers (optional judge model)"),
    State.DETECTING: StateInfo("Detecting (3.3)", activity="pairwise and per-item disagreement metrics"),
    State.RESOLVING: StateInfo("Resolving (Debate)", implemented=False,
                               activity="PLANNED: models debate conflicting answers"),
    State.RESULTS_STORED: StateInfo("Results Stored", activity="results written to the run folder (D3)"),
    State.GENERATING_REPORT: StateInfo("Generating Report (P4)", activity="compile dashboard data and export files"),
    State.REPORT_DELIVERED: StateInfo("Report Delivered", activity="results available to the Researcher"),
    State.FINAL: StateInfo("Final", "final"),
}


@dataclass(frozen=True)
class Transition:
    source: State
    event: str
    guard: str                   # '' = unconditional
    target: State
    tag: str = ""                # short guard text drawn on the diagram
    implemented: bool = True


S = State
TRANSITIONS: tuple[Transition, ...] = (
    Transition(S.INITIAL, "start", "", S.AWAITING_INPUT),
    Transition(S.AWAITING_INPUT, "submitInput", "", S.CONFIGURING),
    Transition(S.CONFIGURING, "configDone", "dataset and settings valid", S.DATASET_READY, "[valid]"),
    Transition(S.CONFIGURING, "configDone", "dataset or settings invalid", S.AWAITING_INPUT, "[invalid]"),
    Transition(S.DATASET_READY, "startAnalysis", "", S.QUERYING),
    Transition(S.QUERYING, "responsesReceived", "at least one model returned usable responses",
               S.NORMALIZING, "[>=1 model ok]"),
    Transition(S.QUERYING, "queryFailed", "no model returned usable responses",
               S.DATASET_READY, "[no model ok]"),
    Transition(S.NORMALIZING, "normalizationDone", "", S.EVALUATING),
    Transition(S.EVALUATING, "evaluationDone", "", S.DETECTING),
    Transition(S.DETECTING, "metricsComputed", "no disagreement detected", S.RESULTS_STORED, "[no disagreement]"),
    # Current behaviour: debate is not implemented, so disagreements go straight to storage.
    Transition(S.DETECTING, "metricsComputed", "disagreement detected, debate not enabled",
               S.RESULTS_STORED, "[disagreement, no debate]"),
    Transition(S.DETECTING, "metricsComputed", "disagreement detected, debate enabled",
               S.RESOLVING, "[disagreement, debate]", implemented=False),
    Transition(S.RESOLVING, "debateConcluded", "", S.RESULTS_STORED, implemented=False),
    Transition(S.RESULTS_STORED, "resultsReady", "", S.GENERATING_REPORT),
    Transition(S.GENERATING_REPORT, "reportCompiled", "", S.REPORT_DELIVERED),
    Transition(S.REPORT_DELIVERED, "newAnalysisRequested", "", S.AWAITING_INPUT),
    Transition(S.REPORT_DELIVERED, "researcherExits", "", S.FINAL),
)


class InvalidTransition(RuntimeError):
    """The pipeline tried to make a move that is not in the STD."""


class StateMachine:
    """Runtime tracker: validates and records every transition of one run."""

    def __init__(self) -> None:
        self.state: State = State.INITIAL
        self.trace: list[dict] = []
        self.failure: dict | None = None
        self._t0 = time.time()

    def fire(self, event: str, guard: str = "", note: str = "") -> State:
        matches = [t for t in TRANSITIONS
                   if t.source == self.state and t.event == event and t.guard == guard]
        if not matches:
            allowed = [f"{t.event}[{t.guard}]" for t in TRANSITIONS if t.source == self.state]
            raise InvalidTransition(
                f"'{event}' [{guard}] is not allowed in {self.state.value}; allowed: {allowed}")
        t = matches[0]
        self.trace.append({"seq": len(self.trace) + 1, "from": t.source.value, "to": t.target.value,
                           "event": t.event, "guard": t.guard, "note": note,
                           "t": round(time.time() - self._t0, 3)})
        self.state = t.target
        return self.state

    def fail(self, message: str) -> None:
        """Record an unexpected error that happened while in the current state."""
        self.failure = {"state": self.state.value, "message": message}

    def to_dict(self) -> dict:
        return {"current": self.state.value, "trace": list(self.trace), "failure": self.failure}


# ----------------------------------------------------------------------------
# Layout (computed from the table, so new states/transitions are placed automatically)
# ----------------------------------------------------------------------------
W, H = 178, 60
COL, ROW = 320, 150
MX, MY = 90, 95
PER_ROW = 4


def _layers() -> dict[State, int]:
    """Longest-path layering after removing back edges found by DFS from INITIAL."""
    adj: dict[State, list[State]] = {s: [] for s in State}
    for t in TRANSITIONS:
        adj[t.source].append(t.target)
    back: set[tuple[State, State]] = set()
    seen: set[State] = set()
    stack: set[State] = set()

    def dfs(u: State) -> None:
        seen.add(u); stack.add(u)
        for v in adj[u]:
            if v in stack:
                back.add((u, v))
            elif v not in seen:
                dfs(v)
        stack.discard(u)

    dfs(State.INITIAL)
    for s in State:                       # states unreachable from INITIAL still get placed
        if s not in seen:
            dfs(s)
    layer = {s: 0 for s in State}
    for _ in range(len(State)):
        for t in TRANSITIONS:
            if (t.source, t.target) not in back:
                layer[t.target] = max(layer[t.target], layer[t.source] + 1)
    return layer


def layout() -> dict[State, tuple[float, float]]:
    """Centre (x, y) of every state. Layers are folded into rows of PER_ROW."""
    layer = _layers()
    return {s: (MX + (l % PER_ROW) * COL + W / 2, MY + (l // PER_ROW) * ROW + H / 2)
            for s, l in layer.items()}


def _half(s: State) -> tuple[float, float]:
    kind = STATES[s].kind
    return (11, 11) if kind == "initial" else (15, 15) if kind == "final" else (W / 2, H / 2)


def _route(t: Transition, pos: dict, back_slot: dict) -> list[tuple[float, float]]:
    """Polyline for one transition (first/last points lie on node borders)."""
    (sx, sy), (tx, ty) = pos[t.source], pos[t.target]
    (shw, shh), (thw, thh) = _half(t.source), _half(t.target)
    layer = _layers()
    forward = layer[t.target] > layer[t.source]
    same_row = abs(sy - ty) < 1
    if same_row and forward:
        if abs(tx - sx) <= COL + 1:
            return [(sx + shw, sy), (tx - thw, ty)]
        d = 30                                                  # skips a node: arc below
        return [(sx + 25, sy + shh), (sx + 25, sy + shh + d), (tx - 25, ty + thh + d), (tx - 25, ty + thh)]
    if same_row:                                                # back edge inside a row: arc above
        d = 34
        return [(sx, sy - shh), (sx, sy - shh - d), (tx + 28, ty - thh - d), (tx + 28, ty - thh)]
    rows = round(abs(ty - sy) / ROW)
    if forward:                                                 # down: elbow through the gap
        def down(x):                                            # forward edges that leave a row downward
            return abs(pos[x.target][1] - pos[x.source][1]) > 1 and layer[x.target] > layer[x.source]
        outs = [x for x in TRANSITIONS if x.source == t.source and down(x)]
        ins = [x for x in TRANSITIONS if x.target == t.target and down(x)]
        k, n = outs.index(t), len(outs)
        j, m = ins.index(t), len(ins)
        top, bottom = sy + shh, ty - thh
        yg = top + (bottom - top) * (0.62 if n == 1 else 0.28 + 0.5 * k / (n - 1))
        sx2, tx2 = sx + (k - (n - 1) / 2) * 22, tx + (j - (m - 1) / 2) * 22
        return [(sx2, sy + shh), (sx2, yg), (tx2, yg), (tx2, ty - thh)]
    if rows == 1:                                               # up one row: second channel in the gap
        top, bottom = ty + thh, sy - shh
        yb = top + (bottom - top) * 0.36
        return [(sx + 30, sy - shh), (sx + 30, yb), (tx - 30, yb), (tx - 30, ty + thh)]
    xr = MX + (PER_ROW - 1) * COL + W + 36                      # up several rows: right margin
    top = MY - 62
    return [(sx + shw, sy), (xr, sy), (xr, top), (tx - 28, top), (tx - 28, ty - thh)]


def _summarise(trace: list[dict], failure: dict | None):
    visits: dict[str, int] = {}
    taken: dict[tuple, int] = {}
    for r in trace:
        visits[r["to"]] = visits.get(r["to"], 0) + 1
        taken[(r["from"], r["event"], r["guard"], r["to"])] = taken.get((r["from"], r["event"], r["guard"], r["to"]), 0) + 1
    if trace:
        visits.setdefault(trace[0]["from"], 1)
    return visits, taken


def diagram_model(sm: StateMachine | dict | None = None) -> dict:
    """JSON-friendly model: every state and transition plus what the run did."""
    d = sm.to_dict() if isinstance(sm, StateMachine) else (sm or {"current": None, "trace": [], "failure": None})
    visits, taken = _summarise(d["trace"], d.get("failure"))
    pos = layout()
    nodes = [{"id": s.value, "label": i.label, "kind": i.kind, "implemented": i.implemented,
              "visits": visits.get(s.value, 0), "current": d["current"] == s.value,
              "failed": bool(d.get("failure") and d["failure"]["state"] == s.value),
              "x": pos[s][0], "y": pos[s][1], "activity": i.activity} for s, i in STATES.items()]
    edges = [{"from": t.source.value, "to": t.target.value, "event": t.event, "guard": t.guard,
              "tag": t.tag, "implemented": t.implemented,
              "taken": taken.get((t.source.value, t.event, t.guard, t.target.value), 0)} for t in TRANSITIONS]
    return {"nodes": nodes, "edges": edges, "trace": d["trace"], "failure": d.get("failure"),
            "current": d["current"]}


# ----------------------------------------------------------------------------
# SVG rendering  (colours come from CSS variables with fallbacks, so it follows the page theme)
# ----------------------------------------------------------------------------
_CSS = """
.n{fill:var(--card,#fff);stroke:var(--sk,#9aa3b5);stroke-width:1.5}
.n.v{fill:var(--okbg,#d9f2e4);stroke:var(--ok,#1f9d63)}
.n.cur{stroke-width:3.5;stroke:var(--ac,#3b6fd8)}
.n.f{fill:var(--wabg,#fbe0df);stroke:var(--wa,#d9534f);stroke-width:3.5}
.n.p{stroke-dasharray:6 4;fill:none}
.t{font:12px system-ui,sans-serif;fill:var(--tx,#1d2433)}.t.p{fill:var(--mu,#667085)}.lb{font:10.5px system-ui,sans-serif;fill:var(--mu,#667085)}
.lb.tk{fill:var(--tx,#1d2433);font-weight:600}.lbbg{fill:var(--bg,#f6f7fb);opacity:.9}
.e{fill:none;stroke:var(--sk,#b8bfce);stroke-width:1.2}.e.tk{stroke:var(--ac,#3b6fd8);stroke-width:2.6}.e.p{stroke-dasharray:5 4}
"""


def _path(pts) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)


def render_svg(sm: StateMachine | dict | None = None) -> str:
    m = diagram_model(sm)
    pos = layout()
    width = MX * 2 + (PER_ROW - 1) * COL + W
    height = MY + ((max(_layers().values())) // PER_ROW) * ROW + H + 80
    back_slot: dict = {}
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" role="img" '
           f'aria-label="State transition diagram"><style>{_CSS}</style>',
           '<defs>'
           '<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
           '<path d="M0,0 L10,5 L0,10 z" fill="#8a93a6"/></marker>'
           '<marker id="art" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto">'
           '<path d="M0,0 L10,5 L0,10 z" fill="#3b6fd8"/></marker></defs>']
    for t, e in zip(TRANSITIONS, m["edges"]):
        pts = _route(t, pos, back_slot)
        cls = "e" + (" tk" if e["taken"] else "") + ("" if e["implemented"] else " p")
        title = f'{t.source.value} --{t.event}{" [" + t.guard + "]" if t.guard else ""}--> {t.target.value}'
        if e["taken"] > 1:
            title += f" (taken {e['taken']}x)"
        out.append(f'<g data-edge="{t.source.value}>{t.target.value}:{t.event}"><title>{escape(title)}</title>'
                   f'<path class="{cls}" d="{_path(pts)}" marker-end="url(#{"art" if e["taken"] else "ar"})"/></g>')
        # label on the longest segment
        seg = max(zip(pts, pts[1:]), key=lambda s: abs(s[0][0] - s[1][0]) + abs(s[0][1] - s[1][1]))
        lx, ly = (seg[0][0] + seg[1][0]) / 2, (seg[0][1] + seg[1][1]) / 2
        taken_x = f" x{e['taken']}" if e["taken"] > 1 else ""
        horizontal = abs(seg[0][1] - seg[1][1]) < 1
        seglen = abs(seg[0][0] - seg[1][0]) + abs(seg[0][1] - seg[1][1])
        one_line = t.event + (f" {t.tag}" if t.tag else "") + taken_x
        cls_l = "lb" + (" tk" if e["taken"] else "")
        if horizontal:
            two = bool(t.tag) and len(one_line) * 5.6 > seglen - 12
            lines = [(t.event + taken_x, ly - 5), (t.tag, ly + 13)] if two else [(one_line, ly - 5)]
            for txt, yy in lines:
                if not txt:
                    continue
                wb = len(txt) * 5.6
                out.append(f'<rect class="lbbg" x="{lx - wb / 2:.1f}" y="{yy - 10:.1f}" width="{wb:.1f}" height="13" rx="3"/>'
                           f'<text class="{cls_l}" x="{lx:.1f}" y="{yy:.1f}" text-anchor="middle">{escape(txt)}</text>')
        else:
            right_side = lx > width / 2
            tx_ = lx - 6 if right_side else lx + 6
            anchor = "end" if right_side else "start"
            wb = len(one_line) * 5.6
            bx = tx_ - wb + 2 if right_side else tx_ - 2
            out.append(f'<rect class="lbbg" x="{bx:.1f}" y="{ly - 10:.1f}" width="{wb:.1f}" height="13" rx="3"/>'
                       f'<text class="{cls_l}" x="{tx_:.1f}" y="{ly:.1f}" text-anchor="{anchor}">{escape(one_line)}</text>')
    for n in m["nodes"]:
        x, y = n["x"], n["y"]
        tip = f'<title>{escape(n["label"])}: {escape(n["activity"]) if n["activity"] else n["kind"]}</title>'
        if n["kind"] == "initial":
            out.append(f'<g data-state="{n["id"]}">{tip}<circle cx="{x}" cy="{y}" r="11" fill="#333"/></g>')
        elif n["kind"] == "final":
            cls = "v" if n["visits"] else ""
            out.append(f'<g data-state="{n["id"]}">{tip}<circle cx="{x}" cy="{y}" r="15" class="n {cls}"/>'
                       f'<circle cx="{x}" cy="{y}" r="8" fill="#333"/></g>')
        else:
            cls = "n" + (" v" if n["visits"] else "") + (" cur" if n["current"] else "") + \
                  (" f" if n["failed"] else "") + ("" if n["implemented"] else " p")
            sub = ""
            if not n["implemented"]:
                sub = f'<text class="lb" x="{x}" y="{y + 17}" text-anchor="middle">planned - not implemented</text>'
            elif n["visits"] > 1:
                sub = f'<text class="lb" x="{x}" y="{y + 17}" text-anchor="middle">visited {n["visits"]}x</text>'
            out.append(f'<g data-state="{n["id"]}">{tip}<rect class="{cls}" x="{x - W / 2}" y="{y - H / 2}" '
                       f'width="{W}" height="{H}" rx="12"/><text class="t{"" if n["implemented"] else " p"}" x="{x}" '
                       f'y="{y - 2 if sub else y + 4}" text-anchor="middle">{escape(n["label"])}</text>{sub}</g>')
    out.append("</svg>")
    return "".join(out)


def to_drawio(sm: StateMachine | dict | None = None) -> str:
    """Editable draw.io file generated from the same table + layout."""
    m = diagram_model(sm)
    pos = layout()
    cells = ['<mxCell id="0"/><mxCell id="1" parent="0"/>']
    for n in m["nodes"]:
        x, y = n["x"], n["y"]
        if n["kind"] == "initial":
            st, w, h, val = "ellipse;shape=startState;fillColor=#000000;", 24, 24, ""
        elif n["kind"] == "final":
            st, w, h, val = "ellipse;shape=endState;fillColor=#000000;", 30, 30, ""
        else:
            fill = "#DAE8FC" if n["implemented"] else "#F5F5F5"
            st = f"rounded=1;arcSize=20;whiteSpace=wrap;html=1;fillColor={fill};strokeColor=#6C8EBF;" + \
                 ("" if n["implemented"] else "dashed=1;")
            w, h = W, H
            val = escape(n["label"] + ("" if n["implemented"] else " (planned)"), {'"': "&quot;"})
        cells.append(f'<mxCell id="{n["id"]}" value="{val}" style="{st}" vertex="1" parent="1">'
                     f'<mxGeometry x="{x - w / 2:.0f}" y="{y - h / 2:.0f}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    back_slot: dict = {}
    for i, t in enumerate(TRANSITIONS):
        pts = _route(t, pos, back_slot)[1:-1]
        wp = "".join(f'<mxPoint x="{x:.0f}" y="{y:.0f}"/>' for x, y in pts)
        label = escape(t.event + (f" [{t.guard}]" if t.guard else ""), {'"': "&quot;"})
        dash = "" if t.implemented else "dashed=1;"
        cells.append(f'<mxCell id="e{i}" value="{label}" style="endArrow=open;html=1;fontSize=10;labelBackgroundColor=#FFFFFF;{dash}" '
                     f'edge="1" parent="1" source="{t.source.value}" target="{t.target.value}"><mxGeometry relative="1" as="geometry">'
                     f'<Array as="points">{wp}</Array></mxGeometry></mxCell>')
    return ('<mxfile host="app.diagrams.net"><diagram name="STD (generated)" id="std-gen">'
            '<mxGraphModel grid="1" gridSize="10" page="1" pageWidth="1400" pageHeight="900"><root>'
            + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
