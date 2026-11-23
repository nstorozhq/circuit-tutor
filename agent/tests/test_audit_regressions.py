import json
import unittest
from pathlib import Path

from app.claims import assess_solution, sourced_solution, uncovered_physics
from app.knowledge import CIRCUITS


class IndependentAuditRegressionTests(unittest.TestCase):
    def test_independent_counterexamples_and_positive_controls(self):
        cases = json.loads(Path(__file__).with_name("audit_contract_cases.json").read_text())
        for case in cases:
            with self.subTest(case=case["id"]):
                solution = sourced_solution(case["claims"], case["text"], CIRCUITS[case["case"]])
                outcome, _ = assess_solution(case["case"], {"step": case["step"]}, solution, case["text"])
                if outcome == "complete" and uncovered_physics(case["text"], solution):
                    outcome = "incomplete"
                self.assertIn(outcome, case["expected_statuses"])

    def test_citation_does_not_validate_an_adjacent_physical_explanation(self):
        text = "R1: A–C, потому что изгиб создаёт два новых узла; R2: C–B; R3: A–B."
        solution = {"terminals": [
            {"element": "R1", "nodes": ["A", "C"], "evidence": "R1: A–C, потому что изгиб создаёт два новых узла"},
            {"element": "R2", "nodes": ["C", "B"], "evidence": "R2: C–B"},
            {"element": "R3", "nodes": ["A", "B"], "evidence": "R3: A–B"},
        ]}
        self.assertTrue(uncovered_physics(text, solution))

    def test_withdrawal_cannot_hide_an_additional_active_claim(self):
        text = "R1: A–B — зачёркиваю, но резистор является узлом. R1: A–C; R2: C–B; R3: A–B."
        solution = {"retracted_evidence": ["R1: A–B — зачёркиваю, но резистор является узлом"], "terminals": [
            {"element": "R1", "nodes": ["A", "C"], "evidence": "R1: A–C"},
            {"element": "R2", "nodes": ["C", "B"], "evidence": "R2: C–B"},
            {"element": "R3", "nodes": ["A", "B"], "evidence": "R3: A–B"},
        ]}
        self.assertTrue(uncovered_physics(text, solution))

    def test_uncertainty_and_recall_are_valid_results(self):
        text = "Выделю все элементы"
        outcome, _ = assess_solution("C03", {"step": "elements", "recalling": True},
                                     {"action": "elements", "action_evidence": text}, text)
        self.assertEqual(outcome, "complete")
        text = "R1: A–C; R2: C–B; R3: A, другой вывод скрыт"
        claims = {"terminals": [
            {"element": "R1", "nodes": ["A", "C"], "evidence": "R1: A–C"},
            {"element": "R2", "nodes": ["C", "B"], "evidence": "R2: C–B"},
            {"element": "R3", "nodes": ["A", "?"], "evidence": "R3: A, другой вывод скрыт"},
        ]}
        claims = sourced_solution(claims, text, CIRCUITS["C06"])
        self.assertEqual(assess_solution("C06", {"step": "nodes"}, claims, text)[0], "complete")
