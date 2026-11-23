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
        data = {"intent": self.intent, "option": "A" if task["choices"] else None, "evidence": text}
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


def protocol_responses(responses):
    import json
    from unittest.mock import AsyncMock
    from app.rag import SUPPORT_PLAN_PROMPT
    first = json.loads(responses[0])
    scope = first.get("help_scope", "procedure")
    pending = iter(responses)
    async def complete(prompt, payload):
        if prompt == SUPPORT_PLAN_PROMPT:
            return json.dumps({"scope": scope, "mode": "explain"})
        return next(pending)
    return AsyncMock(side_effect=complete)


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
        model.complete = protocol_responses([
            '{"intent":"answer","option":"' + correct + '","evidence":"model reconstruction","branches":[]}',
            '{"verdict":"contradiction"}',
            '{"explanation":"Проверь, есть ли ответвления.","source_ids":["C01-series-hint"]}',
            '{"supported":true,"answers_question":true,"reveals_solution":false}',
        ])
        text = "Все три резистора последовательны."
        result, status = asyncio.run(model.rag(text, task, retrieve(CASES["C01"], task, text)))
        self.assertEqual(status, "matched")
        self.assertIsNone(result.option)
        self.assertFalse(result.answer_verified)

    def test_question_uses_concept_sources_and_grounded_explanation(self):
        from unittest.mock import AsyncMock
        from app.model import GigaChat
        from app.knowledge import question, CASES
        from app.rag import retrieve
        model = GigaChat()
        model.credentials = "fixture"
        model.complete = protocol_responses([
            '{"intent":"question","question_text":"А изгиб считается?","help_scope":"concept"}',
            '{"explanation":"Изгиб провода не создаёт нового узла.","source_ids":["concept-0"]}',
            '{"supported":true,"answers_question":true,"reveals_solution":false}',
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
        model.complete = protocol_responses([
            '{"intent":"question","question_text":"А изгиб считается?","help_scope":"concept"}',
            '{"explanation":"Ответ из постороннего файла","source_ids":["outside-lesson"]}',
        ])
        task = question(CASES["C01"], "nodes")
        result, status = asyncio.run(model.rag("Почему?", task, retrieve(CASES["C01"], task, "Почему?")))
        self.assertEqual(status, "matched")
        self.assertFalse(result.answer_verified)
        self.assertNotIn("outside-lesson", result.source_ids)


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


class VerifiedAnswerPriorityTests(unittest.TestCase):
    def test_verified_answer_survives_model_misrouting_and_failure(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        samples = (
            "Привет\nр1 и р2 верх\nр3 нижняя",
            "Здравствуйте, R1 и R2 сверху, R3 внизу",
            "R1 на верхней R2 на верхней R3 на нижней",
        )
        for text in samples:
            for intent, status in (("clarify", "matched"), ("unsure", "matched"), (None, "unavailable")):
                with self.subTest(text=text, intent=intent, status=status):
                    model = SimpleNamespace(enabled=True, rag=AsyncMock(return_value=(
                        RagResult(intent=intent) if intent else None, status,
                    )))
                    state = new_state()
                    reply = asyncio.run(respond(state, text, model))
                    self.assertTrue(reply.startswith("Верно."))
                    self.assertEqual(state["phase"], "guided")
                    self.assertEqual(state["operation"], 1)
                    self.assertEqual(model.rag.await_count, 1)

    def test_definition_keeps_step_and_pending_answer(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from app.rag import retrieve
        from app.knowledge import CASES
        state = lesson_state("C01", 2)
        state["pending"] = {"key": "correct", "prompt": current_question(state)["prompt"]}
        original = copy.deepcopy(state)
        text = "а что такое резистор?"
        sources = retrieve(CASES["C01"], current_question(state), text)
        from app.rag import support_sources
        self.assertTrue(all(source["id"].startswith("concept-") for source in support_sources(sources, text, "concept")))
        source = next(source for source in sources if source["id"] == "concept-6")
        model = SimpleNamespace(enabled=True, rag=AsyncMock(return_value=(
            RagResult(intent="question", explanation=source["text"], source_ids=[source["id"]]), "matched",
        )))
        reply = asyncio.run(respond(state, text, model))
        self.assertIn("Резистор — элемент", reply)
        for field in ("phase", "operation", "index", "scores", "pending"):
            self.assertEqual(state[field], original[field])
