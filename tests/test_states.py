"""STD tests: the diagram is derived from one table and every valid state/transition must appear."""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.states import (STATES, TRANSITIONS, InvalidTransition, State, StateMachine, diagram_model, layout,
                        render_svg, to_drawio)

DOCS = Path(__file__).resolve().parents[1] / "docs" / "STD_LLM_Disagreement_Analysis.drawio"


def run_to(sm, steps):
    for ev, guard in steps:
        sm.fire(ev, guard)
    return sm


OK = [("start", ""), ("submitInput", ""), ("configDone", "dataset and settings valid"), ("startAnalysis", "")]


class TestTable(unittest.TestCase):
    def test_every_state_enum_is_defined_and_used(self):
        self.assertEqual(set(STATES), set(State))
        used = {t.source for t in TRANSITIONS} | {t.target for t in TRANSITIONS}
        self.assertEqual(used, set(State), "a state with no transition can never be reached or left")

    def test_every_state_reachable_from_initial(self):
        seen, todo = {State.INITIAL}, [State.INITIAL]
        while todo:
            for t in TRANSITIONS:
                if t.source == todo[-1] and t.target not in seen:
                    seen.add(t.target); todo.append(t.target); break
            else:
                todo.pop()
        self.assertEqual(seen, set(State))

    def test_non_terminal_states_have_an_exit_and_terminal_has_none(self):
        for s in State:
            outs = [t for t in TRANSITIONS if t.source == s]
            self.assertEqual(not outs, s == State.FINAL, s)

    def test_transitions_unique_per_source_event_guard(self):
        keys = [(t.source, t.event, t.guard) for t in TRANSITIONS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_planned_states_only_reachable_through_planned_transitions(self):
        for s, info in STATES.items():
            if not info.implemented:
                for t in TRANSITIONS:
                    if t.target == s:
                        self.assertFalse(t.implemented, f"{t} leads into unimplemented {s}")


class TestMachine(unittest.TestCase):
    def test_success_path_is_recorded_in_full(self):
        sm = run_to(StateMachine(), OK + [("responsesReceived", "at least one model returned usable responses"),
                    ("normalizationDone", ""), ("evaluationDone", ""),
                    ("metricsComputed", "disagreement detected, debate not enabled"),
                    ("resultsReady", ""), ("reportCompiled", "")])
        self.assertEqual(sm.state, State.REPORT_DELIVERED)
        self.assertEqual([t["to"] for t in sm.trace][0], "AWAITING_INPUT")       # INITIAL is not silently dropped
        self.assertEqual(sm.trace[0]["from"], "INITIAL")
        self.assertEqual(len(sm.trace), 10)

    def test_invalid_transitions_are_refused(self):
        sm = StateMachine()
        with self.assertRaises(InvalidTransition):
            sm.fire("startAnalysis")                                  # not allowed from INITIAL
        run_to(sm, OK)
        with self.assertRaises(InvalidTransition):
            sm.fire("metricsComputed", "no disagreement detected")    # skipping stages
        with self.assertRaises(InvalidTransition):
            sm.fire("responsesReceived", "wrong guard text")

    def test_error_loops_are_kept_as_transitions(self):
        sm = run_to(StateMachine(), OK + [("queryFailed", "no model returned usable responses"), ("startAnalysis", "")])
        self.assertEqual([t["event"] for t in sm.trace][-2:], ["queryFailed", "startAnalysis"])
        self.assertEqual(diagram_model(sm)["nodes"][[n["id"] for n in diagram_model(sm)["nodes"]].index("DATASET_READY")]["visits"], 2)

    def test_invalid_config_returns_to_awaiting_input(self):
        sm = run_to(StateMachine(), [("start", ""), ("submitInput", ""), ("configDone", "dataset or settings invalid")])
        self.assertEqual(sm.state, State.AWAITING_INPUT)

    def test_failure_marks_the_failing_state(self):
        sm = run_to(StateMachine(), OK)
        sm.fail("boom")
        m = diagram_model(sm)
        self.assertTrue(next(n for n in m["nodes"] if n["id"] == "QUERYING")["failed"])
        self.assertIn("boom", render_svg(sm) + str(m["failure"]))


class TestRendering(unittest.TestCase):
    def _check(self, sm):
        svg = render_svg(sm)
        for s in State:
            self.assertIn(f'data-state="{s.value}"', svg, f"{s} missing from the SVG")
        for t in TRANSITIONS:
            self.assertIn(f'data-edge="{t.source.value}&gt;{t.target.value}:{t.event}"'.replace("&gt;", ">"), svg)
        return svg

    def test_blank_diagram_draws_all_states_and_transitions(self):
        svg = self._check(None)
        self.assertEqual(svg.count("data-state="), len(State))
        self.assertEqual(svg.count("data-edge="), len(TRANSITIONS))

    def test_all_runs_draw_the_same_complete_diagram(self):
        for steps in ([], OK, OK + [("queryFailed", "no model returned usable responses")],
                      [("start", ""), ("submitInput", ""), ("configDone", "dataset or settings invalid")]):
            self._check(run_to(StateMachine(), steps))

    def test_unvisited_and_planned_are_distinguished(self):
        svg = render_svg(run_to(StateMachine(), OK))
        self.assertIn("planned - not implemented", svg)
        m = diagram_model(run_to(StateMachine(), OK))
        visited = {n["id"] for n in m["nodes"] if n["visits"]}
        self.assertEqual(visited, {"INITIAL", "AWAITING_INPUT", "CONFIGURING", "DATASET_READY", "QUERYING"})

    def test_layout_gives_every_state_a_distinct_position(self):
        pos = layout()
        self.assertEqual(set(pos), set(State))
        self.assertEqual(len(set(pos.values())), len(State))

    def test_drawio_export_contains_every_state_and_transition(self):
        x = to_drawio(None)
        for s in State:
            self.assertIn(f'id="{s.value}"', x)
        self.assertEqual(x.count('edge="1"'), len(TRANSITIONS))


class TestHandDrawnStd(unittest.TestCase):
    """The lab-notation diagram in docs/ must not drift from the table again."""

    def test_events_and_states_present(self):
        import xml.etree.ElementTree as ET
        values = [c.get("value", "") for c in ET.parse(DOCS).iter("mxCell")]      # real XML parse
        text = re.sub(r"<[^>]+>", " ", " ".join(values))                          # drop the HTML inside labels
        for ev in {t.event for t in TRANSITIONS if t.event != "start"}:
            self.assertIn(ev, text, f"event {ev} missing from the hand-drawn STD")
        for stem in ["Awaiting Input", "Configuring", "Dataset Ready", "Querying", "Normalizing", "Evaluating",
                     "Detecting", "Resolving", "Results Stored", "Generating Report", "Report Delivered"]:
            self.assertIn(stem, text, f"state '{stem}' missing from the hand-drawn STD")
        for t in TRANSITIONS:
            if t.guard and t.implemented:
                self.assertIn(t.guard.split(",")[0].split(" and ")[0][:25], text, f"guard of {t.event} missing")


if __name__ == "__main__":
    unittest.main()
