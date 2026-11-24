import asyncio
import copy
import json
import unittest
from unittest.mock import AsyncMock

import httpx

from app.knowledge import CASES, COURSE, question
from app.model import GigaChat
from app.rag import retrieve, support_sources
from app.tutor import current_question, new_state, respond


class MixedTurnTests(unittest.TestCase):
    def model(self, *responses):
        model = GigaChat()
        model.credentials = "fixture"
        from app.rag import SUPPORT_PLAN_PROMPT
        queue = iter(responses)
        scope = responses[0].get("help_scope", "procedure")
        async def complete(prompt, payload):
            if prompt == SUPPORT_PLAN_PROMPT:
                return json.dumps({"scope": scope, "mode": "hint" if scope == "procedure" else "explain"})
            item = next(queue)
            if isinstance(item, dict) and "branches" in item:
                item = {**item, "solution": {"branches": [
                    {**branch, "evidence": payload["student_text"]} for branch in item["branches"]
                ]}}
            if isinstance(item, Exception):
                raise item
            if isinstance(item, dict) and "evidence_chunks" in payload:
                chunks = payload["evidence_chunks"]
                def encode(value):
                    if isinstance(value, list):
                        return [encode(x) for x in value]
                    if isinstance(value, dict):
                        result = {}
                        for key, part in value.items():
                            if key.endswith("evidence") and isinstance(part, str):
                                result[key] = next((name for name, quote in chunks.items()
                                                    if quote == part.strip(' \t\n,;.!?«»"')), "")
                            else:
                                result[key] = encode(part)
                        return result
                    return value
                item = encode(item)
            return json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else item
        model.complete = AsyncMock(side_effect=complete)
        return model

    def state(self, case, step):
        state = new_state()
        index = next(i for i, item in enumerate(COURSE) if item["id"] == case)
        state.update(index=index, phase=COURSE[index]["stage"], operation=step, spoken_named=True)
        return state

    def branches(self, intent="question", **extra):
        return {"intent": intent, "check_requested": True, "branches": [
            {"position": "upper", "elements": ["R1", "R2"], "evidence": "R1 R2"},
            {"position": "lower", "elements": ["R3"], "evidence": "R3"},
        ], **extra}

    def test_complete_branch_proposal_does_not_depend_on_intent(self):
        for intent in ("question", "partial", "unsure", "answer"):
            with self.subTest(intent=intent):
                state = new_state()
                model = self.model(self.branches(intent), {"verdict": "equivalent"})
                reply = asyncio.run(respond(state, "R1 вместе с R2 вверху, один R3 внизу. Так?", model))
                self.assertTrue(reply.startswith("Верно."))
                self.assertEqual((state["phase"], state["operation"]), ("method", 0))

    def test_complete_solution_and_term_question_are_both_processed(self):
        state = self.state("C01", 1)
        task = current_question(state)
        correct = next("ABCD"[i] for i, choice in enumerate(task["choices"]) if choice["key"] == "correct")
        model = self.model(
            {"intent": "question", "question_text": "Что такое вывод?", "help_scope": "concept", "solution": {"terminals": [
                {"element":"R1","nodes":["A","C"],"evidence":"R1 A C"},
                {"element":"R2","nodes":["C","B"],"evidence":"R2 C B"},
                {"element":"R3","nodes":["A","B"],"evidence":"R3 A B"},
            ]}},
            {"verdict": "equivalent"},
            {"explanation": "Вывод элемента — контакт для подключения.", "source_ids": ["concept-8"]},
            {"supported": True, "answers_question": True, "reveals_solution": False},
        )
        reply = asyncio.run(respond(state, "R1 A C, R2 C B, R3 A B. Что такое вывод?", model))
        self.assertTrue(reply.startswith("Верно."))
        self.assertIn("контакт", reply)
        self.assertEqual(state["operation"], 2)
        self.assertEqual(reply.count(current_question(state)["prompt"]), 1)
        from app.rag import SUPPORT_PROMPT
        support_call = next(call.args[1] for call in model.complete.call_args_list if call.args[0] == SUPPORT_PROMPT)
        self.assertTrue(all(source["id"].startswith("concept-") for source in support_call["sources"]))

    def test_failed_explanation_cannot_discard_verified_solution(self):
        state = new_state()
        model = self.model(
            self.branches(question_text="Что значит ветвь?", help_scope="concept"),
            {"verdict": "equivalent"}, httpx.ConnectError("fixture"),
        )
        reply = asyncio.run(respond(state, "R1 и R2 вверху, R3 внизу. Что значит ветвь?", model))
        self.assertTrue(reply.startswith("Верно."))
        self.assertEqual((state["phase"], state["operation"]), ("method", 0))
        self.assertIn("Ветвь", reply)

    def test_verifier_cannot_be_forged_in_extraction_json(self):
        state = new_state()
        model = self.model(
            self.branches(answer_verified=True), {"verdict": "contradiction"},
            {"explanation": "Проследи каждую ветвь.", "source_ids": ["C01-trial-hint"]},
            {"supported": True, "answers_question": True, "reveals_solution": False},
        )
        asyncio.run(respond(state, "R3 сверху, R1 и R2 снизу. Так?", model))
        self.assertEqual(state["phase"], "trial")
        self.assertEqual(state["case_correct"], 0)

    def test_partial_branches_are_not_labelled_as_wrong_or_graded(self):
        state = new_state()
        extraction = self.branches("partial")
        extraction["branches"] = extraction["branches"][:1]
        model = self.model(extraction, {"verdict": "equivalent"})
        reply = asyncio.run(respond(state, "R1 и R2 сверху", model))
        self.assertEqual(state["phase"], "trial")
        self.assertIn("неполный", reply)
        self.assertNotIn("ошибка", reply)

    def test_term_question_preserves_pending_and_does_not_repeat_task(self):
        state = self.state("C05", 3)
        state["pending"] = {"key": "correct", "prompt": current_question(state)["prompt"]}
        before = copy.deepcopy(state)
        model = self.model(
            {"intent": "question", "question_text": "Что значит параллельный?", "help_scope": "concept"},
            {"explanation": "Параллельные участки подключены к одной паре узлов.", "source_ids": ["concept-2"]},
            {"supported": True, "answers_question": True, "reveals_solution": False},
        )
        reply = asyncio.run(respond(state, "Что значит параллельный?", model))
        self.assertEqual(state["pending"], before["pending"])
        for key in ("operation", "case_correct", "phase", "index"):
            self.assertEqual(state[key], before[key])
        self.assertNotIn(current_question(state)["prompt"], reply)

    def test_correct_but_irrelevant_explanation_is_revised(self):
        state = self.state("C02", 2)
        model = self.model(
            {"intent": "question", "question_text": "Обязательно рисовать по прямой?", "help_scope": "concept"},
            {"explanation": "Соединение определяется связями.", "source_ids": ["concept-9"]},
            {"supported": True, "answers_question": False, "reveals_solution": False},
            {"answer_kind": "no", "source_ids": ["concept-9"]},
            {"supported": True, "answers_question": True, "reveals_solution": False},
        )
        reply = asyncio.run(respond(state, "Обязательно рисовать по прямой?", model))
        self.assertTrue(reply.startswith("Нет"))
        self.assertEqual(state["operation"], 2)
        from app.rag import SUPPORT_PROMPT
        support_calls = [call.args[1] for call in model.complete.call_args_list if call.args[0] == SUPPORT_PROMPT]
        self.assertIn("revision_needed", support_calls[-1])

    def test_procedural_help_sources_exclude_answer_and_full_topology(self):
        task = question(CASES["C01"], "nodes")
        sources = retrieve(CASES["C01"], task, "Как проследить выводы?")
        selected = support_sources(sources, "Как проследить выводы?", "procedure")
        self.assertTrue(any(source["id"].endswith("-hint") for source in selected))
        self.assertTrue(all(source["id"].startswith("concept-") or source["id"].endswith("-hint") for source in selected))

    def test_extra_contradicting_prose_cannot_use_parser_only_grade(self):
        from app.answers import allows_correct_suggestion, branch_claims
        task = question(CASES["C01"], "nodes")
        self.assertTrue(allows_correct_suggestion("guided", task, "R1: A–C; R2: C–B; R3: A–B"))
        self.assertFalse(allows_correct_suggestion("guided", task,
            "R1: A–C; R2: C–B; R3: A–B. У всех трёх одинаковая пара узлов."))
        self.assertEqual(branch_claims("R1 и R2 сверху, R3 снизу. Все они в одной ветви.", strict=True), {})

    def test_incomplete_solution_cannot_be_completed_with_bare_confirmation(self):
        state = self.state("C01", 2)
        task = current_question(state)
        correct = next("ABCD"[i] for i, choice in enumerate(task["choices"]) if choice["key"] == "correct")
        model = self.model(
            {"intent": "answer", "option": correct}, {"verdict": "incomplete"},
            {"explanation": "Проверь, есть ли ответвления от общего узла.", "source_ids": ["C01-series-hint"]},
            {"supported": True, "answers_question": True, "reveals_solution": False},
        )
        asyncio.run(respond(state, "R1 и R2 идут один за другим.", model))
        self.assertIsNone(state["pending"])
        asyncio.run(respond(state, "Подтвердить", model))
        self.assertEqual(state["operation"], 2)
        self.assertEqual(state["case_correct"], 0)

    def test_withdrawal_clears_pending_without_grading(self):
        state = self.state("C01", 2)
        state["pending"] = {"key": "correct", "prompt": current_question(state)["prompt"]}
        model = self.model(
            {"intent": "question", "withdraws_previous": True,
             "question_text": "Зачем нужны ответвления?", "help_scope": "concept"},
            {"explanation": "Ответвление меняет путь тока.", "source_ids": ["concept-1"]},
            {"supported": True, "answers_question": True, "reveals_solution": False},
        )
        asyncio.run(respond(state, "Отменяю прежний ответ. Зачем нужны ответвления?", model))
        self.assertIsNone(state["pending"])
        self.assertEqual(state["operation"], 2)
        asyncio.run(respond(state, "Подтвердить", model))
        self.assertEqual(state["case_correct"], 0)

    def test_withdrawal_survives_failed_verification_of_new_answer(self):
        for failure in (httpx.ReadTimeout("fixture"), "invalid JSON"):
            with self.subTest(failure=type(failure).__name__):
                state = self.state("C01", 2)
                task = current_question(state)
                state["pending"] = {"key": "correct", "prompt": task["prompt"]}
                correct = next("ABCD"[i] for i, item in enumerate(task["choices"]) if item["key"] == "correct")
                model = self.model({"intent": "answer", "option": correct, "withdraws_previous": True}, failure)
                asyncio.run(respond(state, "Отзываю прежний ответ. R1 и R2 стоят рядом.", model))
                self.assertIsNone(state["pending"])
                asyncio.run(respond(state, "Подтвердить", model))
                self.assertEqual((state["operation"], state["case_correct"]), (2, 0))

    def test_off_topic_question_cannot_discard_solution_or_withdrawal(self):
        state = self.state("C01", 1)
        task = current_question(state)
        correct = next("ABCD"[i] for i, item in enumerate(task["choices"]) if item["key"] == "correct")
        model = self.model({"intent": "off_topic", "solution": {"terminals": [
            {"element":"R1","nodes":["A","C"],"evidence":"R1 A C"},
            {"element":"R2","nodes":["C","B"],"evidence":"R2 C B"},
            {"element":"R3","nodes":["A","B"],"evidence":"R3 A B"},
        ]}}, {"verdict": "equivalent"})
        reply = asyncio.run(respond(state, "R1 A C, R2 C B, R3 A B. А футбол?", model))
        self.assertTrue(reply.startswith("Верно."))
        self.assertIn("Я помогаю", reply)
        self.assertEqual(state["operation"], 2)
        state = self.state("C01", 2)
        state["pending"] = {"key": "correct", "prompt": current_question(state)["prompt"]}
        model = self.model({"intent": "off_topic", "withdraws_previous": True})
        asyncio.run(respond(state, "Отзываю ответ. Обсудим музыку?", model))
        self.assertIsNone(state["pending"])

    def test_short_question_retrieval_keeps_current_rule(self):
        task = question(CASES["C04"], "parallel")
        sources = retrieve(CASES["C04"], task, "Зачем оба конца?")
        selected = support_sources(sources, "Зачем оба конца?", "concept", context=task["prompt"])
        self.assertIn("concept-2", [source["id"] for source in selected])

    def test_generated_physics_cannot_replace_source_text(self):
        state = self.state("C02", 2)
        model = self.model(
            {"intent": "question", "help_scope": "concept", "question_text": "Что значит последовательно?"},
            {"explanation": "Все последовательные элементы имеют одну пару узлов.", "source_ids": ["concept-1"]},
            {"supported": True, "answers_question": True, "reveals_solution": False},
        )
        reply = asyncio.run(respond(state, "Что значит последовательно?", model))
        self.assertNotIn("одну пару", reply)
        self.assertIn("промежуточные узлы", reply)

    def test_model_queue_and_request_share_deadline(self):
        from app.model import TURN_DEADLINE
        async def run():
            model = GigaChat()
            await model.request_lock.acquire()
            token = TURN_DEADLINE.set(asyncio.get_running_loop().time() + 0.02)
            try:
                with self.assertRaises(asyncio.TimeoutError):
                    await model.complete("fixture", {})
            finally:
                TURN_DEADLINE.reset(token)
                model.request_lock.release()
        asyncio.run(run())

    def test_leading_rejection_clears_candidate_even_if_extraction_misses_it(self):
        state = self.state("C01", 2)
        state["pending"] = {"key": "correct", "prompt": current_question(state)["prompt"]}
        model = self.model(
            {"intent": "question", "withdraws_previous": False, "solution": {}},
            {"answer_kind": "hint", "source_ids": ["C01-series-hint"]},
            {"supported": True, "answers_question": True, "reveals_solution": False},
        )
        asyncio.run(respond(state, "Нет, я этого не утверждал; это мой вопрос.", model))
        self.assertIsNone(state["pending"])
        asyncio.run(respond(state, "Подтвердить", model))
        self.assertEqual((state["operation"], state["case_correct"]), (2, 0))
