import asyncio
import copy
import unittest

from app.knowledge import COURSE
from app.rag import RagResult, validate_result
from app.tutor import current_question, new_state, respond


def lesson_state(case_id, step=0, *, naming=False):
    state = new_state()
    index = next(i for i, case in enumerate(COURSE) if case["id"] == case_id)
    state.update(index=index, phase=COURSE[index]["stage"], operation=step,
                 spoken_named=not naming)
    return state


class DialogueModel:
    enabled = True

    def __init__(self, intent):
        self.intent = intent
        self.history = None
        self.calls = 0

    async def rag(self, text, task, sources, *, history=()):
        self.history = history
        self.calls += 1
        data = {"intent": self.intent, "option": "A", "evidence": text}
        result = validate_result(RagResult.model_validate(data), text, sources, task["choices"])
        return result, "matched"


class DialoguePolicyTests(unittest.TestCase):
    def test_non_answers_never_grade_or_advance_any_training_stage(self):
        states = [new_state(), lesson_state("C01", 1), lesson_state("C02", 2),
                  lesson_state("C03", 1, naming=True), lesson_state("C05", 3),
                  lesson_state("C07"), lesson_state("C08")]
        for state in states:
            for intent in ("unsure", "hint", "question", "clarify", "partial", "off_topic"):
                with self.subTest(stage=state["phase"], operation=state["operation"], intent=intent):
                    current = copy.deepcopy(state)
                    model = DialogueModel(intent)
                    message = asyncio.run(respond(current, "Можешь помочь разобраться с этим шагом?", model))
                    for field in ("phase", "index", "operation", "spoken_named", "case_correct", "scores"):
                        self.assertEqual(current[field], state[field])
                    self.assertIsNone(current["pending"])
                    self.assertTrue(message)
                    self.assertNotIn("Не удалось проверить", message)
                    self.assertEqual(model.calls, 1)

    def test_question_preserves_pending_answer_and_receives_recent_dialogue(self):
        state = lesson_state("C01", 2)
        task = current_question(state)
        state["pending"] = {"key": "correct", "prompt": task["prompt"]}
        history = [{"role": "user", "text": "Я рассматриваю верхнюю ветвь."},
                   {"role": "assistant", "text": "Уточни, какие элементы ты имеешь в виду."}]
        model = DialogueModel("question")
        asyncio.run(respond(state, "А почему?", model, history=history))
        self.assertEqual(model.history, history)
        self.assertEqual(state["pending"]["key"], "correct")
        asyncio.run(respond(state, "Подтвердить", model))
        self.assertEqual(state["operation"], 3)

    def test_control_requests_never_retrieve_or_reveal_an_answer(self):
        for text in ("Теряюсь, объясни ещё раз", "А почему здесь общий узел?", "Подсказка"):
            state = lesson_state("C09")
            model = DialogueModel("hint")
            message = asyncio.run(respond(state, text, model))
            self.assertEqual(model.calls, 0)
            self.assertEqual(state["operation"], 0)
            self.assertEqual(state["scores"], {})
            self.assertNotIn("L1: A", message)


class ModelProtocolTests(unittest.TestCase):
    def test_wrong_semantic_match_cannot_be_proposed_as_correct(self):
        from unittest.mock import AsyncMock
        from app.model import GigaChat
        from app.knowledge import question, CASES
        from app.rag import retrieve
        task = question(CASES["C01"], "series")
        correct = next("ABCD"[i] for i, choice in enumerate(task["choices"]) if choice["key"] == "correct")
        model = GigaChat()
        model.credentials = "fixture"
        model.complete = AsyncMock(side_effect=[
            '{"intent":"answer"}',
            '{"option":"' + correct + '","evidence":"model reconstruction","branches":[]}',
            '{"verdict":"contradiction"}',
        ])
        text = "Все три резистора последовательны."
        result, status = asyncio.run(model.rag(text, task, retrieve(CASES["C01"], task, text)))
        self.assertEqual(status, "matched")
        self.assertIsNone(result.option)
        self.assertEqual(result.intent, "hint")

    def test_plain_intent_and_grounded_explanation_are_supported(self):
        from unittest.mock import AsyncMock
        from app.model import GigaChat
        from app.knowledge import question, CASES
        from app.rag import retrieve
        model = GigaChat()
        model.credentials = "fixture"
        model.complete = AsyncMock(side_effect=[
            "question",
            '{"explanation":"Изгиб провода не создаёт нового узла.","source_ids":["concept-0"]}',
            '{"verdict":"supported"}',
        ])
        task = question(CASES["C01"], "nodes")
        text = "А изгиб считается?"
        sources = retrieve(CASES["C01"], task, text)
        self.assertIn("concept-0", [source["id"] for source in sources])
        result, status = asyncio.run(model.rag(text, task, sources))
        self.assertEqual(status, "matched")
        self.assertEqual(result.intent, "question")
        self.assertIn("Изгиб", result.explanation)

    def test_unknown_source_in_explanation_is_rejected(self):
        from unittest.mock import AsyncMock
        from app.model import GigaChat
        from app.knowledge import question, CASES
        from app.rag import retrieve
        model = GigaChat()
        model.credentials = "fixture"
        model.complete = AsyncMock(side_effect=[
            "question",
            '{"explanation":"Ответ из постороннего файла","source_ids":["outside-lesson"]}',
        ])
        task = question(CASES["C01"], "nodes")
        result, status = asyncio.run(model.rag("Почему?", task, retrieve(CASES["C01"], task, "Почему?")))
        self.assertIsNone(result)
        self.assertEqual(status, "unclear")


class CircuitClaimTests(unittest.TestCase):
    def test_quantified_claims_use_the_actual_circuit_topology(self):
        from app.answers import contradicts_all_elements
        from app.knowledge import CIRCUITS
        for case_id, step, conflict in (
            ("C01", "series", True), ("C02", "series", False),
            ("C04", "series", True), ("C03", "parallel", False),
            ("C05", "parallel", True), ("C07", "parallel", False),
        ):
            with self.subTest(case=case_id, step=step):
                self.assertEqual(
                    contradicts_all_elements("Все элементы соединены одинаково", step, CIRCUITS[case_id]),
                    conflict,
                )

    def test_ordinal_references_are_resolved_against_current_elements(self):
        from app.answers import node_claims, resolve_ordinal_elements
        from app.knowledge import CIRCUITS
        shape = CIRCUITS["C01"]
        claims = node_claims(resolve_ordinal_elements(
            "У первого концы A и C, у второго C и B, у третьего A и B", shape.components,
        ))
        self.assertEqual(claims, {
            name.lower(): frozenset((a.lower(), b.lower())) for name, a, b in shape.components
        })
