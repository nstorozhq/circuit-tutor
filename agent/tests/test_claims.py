import unittest

from app.claims import assess_solution, evidence_chunks, resolve_evidence, solution_schema, sourced_solution, uncovered_physics
from app.knowledge import CASES, CIRCUITS, question
from app.rag import retrieve, support_sources


class PhysicalEvidenceTests(unittest.TestCase):
    def test_literal_pairs_restore_model_slots_without_losing_conflicts(self):
        text = "R1: C–A; R2: B–C; R3: B–A. А буква R обозначает резистор?"
        claims = {"terminals": [{"element": part.split(":")[0], "nodes": [], "evidence": part}
                                for part in text.split(". ")[0].split("; ")]}
        solution = sourced_solution(claims, text, CIRCUITS["C01"])
        self.assertEqual(self.assess("C01", "nodes", text, **solution), "complete")
        text += " R3: A–C"
        claims["terminals"].append({"element": "R3", "nodes": [], "evidence": "R3: A–C"})
        solution = sourced_solution(claims, text, CIRCUITS["C01"])
        self.assertEqual(self.assess("C01", "nodes", text, **solution), "contradiction")

    def test_literal_pair_repair_does_not_swallow_conditions_or_false_prose(self):
        text = "R1: C–A; R2: B–C; R3: B–A. Все три подключены одинаково."
        claims = {"terminals": [{"element": part.split(":")[0], "nodes": [], "evidence": part}
                                for part in text.split(". ")[0].split("; ")]}
        solution = sourced_solution(claims, text, CIRCUITS["C01"])
        self.assertTrue(uncovered_physics(text, solution))
        for quote in ("Не R1: C–A", "Если R1: C–A", "R1: C–A, потому что изгиб создаёт узел"):
            result = sourced_solution({"terminals": [{"element": "R1", "nodes": [], "evidence": quote}]},
                                      quote, CIRCUITS["C01"])
            self.assertTrue(all(not item["nodes"] for item in result["terminals"]))

    def assess(self, case, step, text, **solution):
        return assess_solution(case, question(CASES[case], step), solution, text)[0]

    def pairs(self, reverse=False):
        values = [("R1", "A", "C"), ("R2", "C", "B"), ("R3", "A", "B")]
        if reverse:
            values = [(name, b, a) for name, a, b in reversed(values)]
        text = "; ".join(f"{name}: {a}–{b}" for name, a, b in values)
        facts = [{"element": name, "nodes": [a, b], "evidence": f"{name}: {a}–{b}"} for name, a, b in values]
        return text, facts

    def test_node_and_element_order_does_not_change_correct_result(self):
        for reverse in (False, True):
            text, facts = self.pairs(reverse)
            self.assertEqual(self.assess("C01", "nodes", text, terminals=facts), "complete")
            self.assertEqual(self.assess("C01", "nodes", text, terminals=facts[:-1]), "incomplete")

    def test_a_false_universal_cannot_be_hidden_behind_correct_pairs(self):
        text, facts = self.pairs()
        claim = "У всех трёх одинаковая пара узлов"
        text += ". " + claim
        self.assertEqual(self.assess("C01", "nodes", text, terminals=facts), "incomplete")
        self.assertEqual(self.assess("C01", "nodes", text, terminals=facts,
            universals=[{"relation": "same_terminals", "value": True, "evidence": claim}]), "contradiction")

    def test_two_active_assignments_for_one_component_conflict(self):
        text, facts = self.pairs()
        text += "; R3: A–C"
        facts.append({"element": "R3", "nodes": ["A", "C"], "evidence": "R3: A–C"})
        self.assertEqual(self.assess("C01", "nodes", text, terminals=facts), "contradiction")

    def test_parallel_nodes_must_be_explicit_in_the_learners_evidence(self):
        text = "Параллельны R1–R2 и R3"
        item = {"relation": "parallel", "groups": ["R1+R2", "R3"], "nodes": ["A", "B"], "evidence": text}
        self.assertEqual(self.assess("C01", "parallel", text, relations=[item]), "incomplete")
        item["evidence"] = text + " между A и B"
        self.assertEqual(self.assess("C01", "parallel", item["evidence"], relations=[item]), "complete")

    def test_one_correct_series_group_does_not_cover_the_other_group(self):
        text = "L1 и L2 соединены последовательно, в C нет ответвления"
        item = {"relation": "series", "groups": ["L1", "L2"], "nodes": ["C"],
                "no_extra_branch": True, "evidence": text}
        self.assertEqual(self.assess("C05", "series", text, relations=[item]), "incomplete")

    def test_series_label_does_not_supply_missing_justification(self):
        text = "R1 и R2 последовательны, общий узел C"
        item = {"relation": "series", "groups": ["R1", "R2"], "nodes": ["C"],
                "no_extra_branch": True, "evidence": text}
        self.assertEqual(self.assess("C01", "series", text, relations=[item]), "incomplete")
        text += ", нет ответвлений"
        item["evidence"] = text
        self.assertEqual(self.assess("C01", "series", text, relations=[item]), "complete")

    def test_false_parallel_relationship_rejected_with_present_nodes(self):
        text = "R1 и R3 параллельны между A и B"
        item = {"relation": "parallel", "groups": ["R1", "R3"], "nodes": ["A", "B"], "evidence": text}
        self.assertEqual(self.assess("C01", "parallel", text, relations=[item]), "contradiction")

    def test_evidence_chunks_and_schema_cannot_assign_whole_answer_to_one_element(self):
        text = "R1 A C, R2 C B, R3 A B. А буква R что значит?"
        chunks = evidence_chunks(text)
        schema = solution_schema("nodes", ["R1", "R2", "R3"], ["A", "B", "C"], chunks)
        for variant in schema["properties"]["terminals"]["items"]["anyOf"]:
            element = variant["properties"]["element"]["enum"][0]
            permitted = variant["properties"]["evidence"]["enum"]
            self.assertTrue(any(chunks[key].startswith(element) for key in permitted if key))
            self.assertNotIn("s0", permitted)
        resolved = resolve_evidence({"evidence": "not-an-id"}, chunks)
        self.assertEqual(resolved["evidence"], "")

    def test_composite_series_facts_keep_the_external_branch_qualification(self):
        for case in ("C04", "C09", "C11"):
            task = question(CASES[case], "series")
            sources = retrieve(CASES[case], task, "Почему эти участки последовательны?")
            selected = support_sources(sources, "Почему эти участки последовательны?", "drawing")
            fact = next(source for source in selected if source["id"] == case + "-relation-series-0")
            self.assertIn("между этими участками нет отдельной ветви", fact["text"])
            self.assertNotIn("Нет ответвлений", fact["text"])

    def test_hidden_series_is_not_presented_as_an_unconditional_fact(self):
        for case in ("C06", "C08"):
            task = question(CASES[case], "series")
            selected = support_sources(retrieve(CASES[case], task, "Почему?"), "Почему?", "drawing")
            self.assertFalse(any(source["id"].startswith(case + "-relation-series-") for source in selected))
