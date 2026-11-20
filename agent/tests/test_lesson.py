import itertools
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from fastapi.testclient import TestClient

from app.knowledge import ACTION_NAMES, CASES, CIRCUITS, STEPS, question
from app.main import create_app


class OfflineModel:
    enabled = False

    async def interpret(self, answer, prompt, choices):
        return None, "disabled"


class FixtureModel:
    enabled = True

    def __init__(self):
        self.options = []

    async def interpret(self, answer, prompt, choices):
        self.options = [option["text"] for option in choices]
        return next(index for index, option in enumerate(choices) if option["key"] == "correct"), "matched"


class LessonTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.db = Path(temporary.name) / "test.sqlite3"
        self.client = self.enterContext(TestClient(create_app(self.db, OfflineModel())))
        self.session = self.client.post("/api/sessions").json()
        self.url = "/api/sessions/" + self.session["id"]
        self.turn_ids = itertools.count(1)

    def send(self, text):
        response = self.client.post(self.url + "/turns", json={
            "message_id": f"test-{next(self.turn_ids)}", "version": self.session["version"], "text": text,
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.session = response.json()
        return self.session

    def choose(self, start):
        return self.send(next(choice for choice in self.session["state"]["choices"] if choice.startswith(start)))

    def enter_training(self):
        self.choose("Определять способ")
        self.send("Я вижу три резистора и разные ветви.")
        self.choose("Мой вывод или обоснование")
        self.choose("Выделить элементы, определить")

    def test_first_attempt_requires_relevant_answer_and_comparison(self):
        opening = self.session["messages"][0]["text"]
        self.assertTrue(opening.startswith("Привет! Я твой помощник по физике."))
        self.assertIn("напиши «подсказка»", opening)
        self.assertIn("Что будем учиться делать по электрической схеме?", opening)
        self.assertNotIn("нет ответвлений", opening)
        hint = self.send("подсказка")
        self.assertEqual(hint["state"]["phase"], "goal")
        self.assertIn("к каким точкам подключён каждый элемент", hint["message"]["text"])
        wrong_goal = self.choose("Вычислять мощность")
        self.assertEqual(wrong_goal["state"]["phase"], "goal")
        self.assertIn("Этот вариант не подходит", wrong_goal["message"]["text"])
        self.assertIn("Для вычисления мощности нужны числовые данные", wrong_goal["message"]["text"])
        self.assertIn("Выбери другой вариант", wrong_goal["message"]["text"])
        self.choose("Определять способ")
        self.assertEqual(self.session["state"]["phase"], "trial")
        self.assertIsNotNone(self.session["state"]["circuit"])
        self.assertIn("нет ответвлений", self.session["message"]["text"])
        result = self.send("На схеме фруктовый салат.")
        self.assertEqual(result["state"]["phase"], "trial")
        result = self.send("Думаю, все три резистора последовательны.")
        self.assertEqual(result["state"]["phase"], "compare")
        self.assertIn("Ответ учителя", result["message"]["text"])
        self.assertIn("Смешанное", result["message"]["text"])
        self.assertIsNotNone(result["state"]["circuit"])
        self.choose("Мой вывод или обоснование")
        self.assertEqual(self.session["state"]["phase"], "method")

    def test_reopens_goal_from_before_question_change(self):
        previous = "Какое действие мы будем тренировать с помощью этих знаний?"
        with closing(sqlite3.connect(self.db)) as db, db:
            state = json.loads(db.execute("SELECT state FROM sessions").fetchone()[0])
            state["pending"] = {"key": "correct", "prompt": previous}
            history = [{"id": "reply-legacy", "role": "assistant", "text": f"Предыдущая реплика.\n\n{previous}"}]
            db.execute("UPDATE sessions SET state = ?, history = ?", (json.dumps(state), json.dumps(history)))
        current = self.client.get(self.url).json()
        self.assertIn(current["state"]["question"], current["messages"][-1]["text"])
        self.assertNotIn(previous, current["messages"][-1]["text"])
        self.assertTrue(current["state"]["pending"])
        self.assertEqual(self.send("Подтвердить")["state"]["phase"], "trial")

    def enter_spoken(self):
        self.enter_training()
        for case_id in ("C01", "C02"):
            for step in STEPS:
                self.send(question(CASES[case_id], step)["reference"])

    def test_spoken_requires_recall_and_reveals_options_only_on_request(self):
        self.enter_spoken()
        initial = self.session["state"]
        self.assertEqual(initial["case_id"], "C03")
        self.assertEqual(initial["choices"], [])
        self.assertTrue(initial["can_reveal_options"])
        self.assertNotIn("элементы", initial["question"].lower())
        result = self.send("L1, L2")
        self.assertEqual(result["state"]["question"], initial["question"])
        result = self.send("Сначала выделяю элементы. L1, L2.")
        self.assertEqual(result["state"]["step_label"], "Узлы")
        self.assertEqual(result["state"]["choices"], [])
        result = self.send("Показать варианты")
        self.assertIn("Определяю узлы подключения.", result["state"]["choices"])
        result = self.send("Определяю узлы подключения.")
        self.assertIn("Выполни названное", result["state"]["question"])
        self.assertEqual(result["state"]["choices"], [])
        result = self.send("L1: B–A; L2: A–B")
        self.assertEqual(result["state"]["step_label"], "Последовательные участки")

    def test_diagnoses_single_common_node_and_waits_for_correction(self):
        self.enter_training()
        self.send("R1, R2, R3")
        self.choose("R1: A–C; R2: C–B; R3: A–B")
        self.choose("R1 и R2 соединены последовательно")
        result = self.choose("R1 и R3 параллельны")
        self.assertEqual(result["state"]["step_label"], "Параллельные участки")
        self.assertIn("подключены к C и B", result["message"]["text"])
        result = self.choose("(R1–R2) и R3")
        self.assertEqual(result["state"]["step_label"], "Вывод")

    def test_reference_walkthrough_and_teacher_mode_leave_current_step(self):
        self.enter_training()
        result = self.send("Покажи ответ")
        self.assertIn("подключены к одним узлам A и B", result["message"]["text"])
        self.assertNotIn("Элементы —", result["message"]["text"])
        result = self.send("Полный разбор")
        self.assertIn("Элементы — R1, R2, R3", result["message"]["text"])
        self.assertEqual(result["state"]["step_label"], "Элементы")
        result = self.send("Разговорный режим")
        self.assertTrue(result["state"]["teacher"])
        self.assertEqual(result["state"]["case_id"], "C01")
        self.assertFalse(self.send("Краткий режим")["state"]["teacher"])

    def test_empty_and_off_topic_stay_on_step(self):
        opening = self.session["state"]
        for text in ("привет", "кто такой Дональд Трамп", "Где находится Париж?"):
            with self.subTest(text=text):
                result = self.send(text)
                self.assertEqual(result["state"], opening)
                self.assertIn("Я помогаю разбирать соединение проводников", result["message"]["text"])
                self.assertIn(opening["question"], result["message"]["text"])
                self.assertNotIn("Укажи элементы, узлы и вывод", result["message"]["text"])
        self.enter_training()
        for text in ("", "Кто выиграл чемпионат?"):
            result = self.send(text)
            self.assertEqual(result["state"]["step_label"], "Элементы")
            self.assertNotIn("R1, R2, R3", result["message"]["text"])

    def test_full_lesson_scores_locked_control_and_wrong_answer(self):
        self.enter_training()
        for index in range(1, 9):
            case = CASES[f"C{index:02}"]
            steps = STEPS if index <= 6 else ("overall",)
            for step in steps:
                if 3 <= index <= 6:
                    self.send(ACTION_NAMES[step])
                self.send(question(case, step)["reference"])
        self.assertEqual(self.session["state"]["phase"], "test")
        result = self.send("Полный разбор C09")
        self.assertIn("после сдачи контрольной", result["message"]["text"])
        self.assertIsNone(result["state"]["scores"])
        for index in (9, 10):
            case = CASES[f"C{index:02}"]
            for step in STEPS[1:]:
                data = question(case, step)
                answer = next(option["text"] for option in data["choices"] if option["key"] != "correct") if (index, step) == (9, "parallel") else data["reference"]
                result = self.send(answer)
                if (index, step) == (9, "parallel"):
                    self.assertIn("Ответ записан", result["message"]["text"])
        self.assertEqual(result["state"]["phase"], "done")
        self.assertEqual(result["state"]["scores"], {"C09": 3, "C10": 4})
        self.assertIn("параллельные участки", result["message"]["text"])
        self.assertIn("Элементы — L1, L2, L3", self.send("Полный разбор C09")["message"]["text"])
        self.assertEqual(self.send("Резерв")["state"]["case_id"], "C11")

    def test_hidden_nodes_are_not_claimed_as_known(self):
        for case_id, hidden in (("C06", "R3"), ("C08", "L3")):
            self.assertIn([hidden, "A", None], CIRCUITS[case_id].components)
            data = CASES[case_id]
            series = question(data, "series")["reference"]
            self.assertIn(f"если скрытый вывод {hidden} не подключён к C", series)
            self.assertIn("там есть ответвление", series)
            self.assertNotIn("Нет ответвлений в узлах C", series)
            self.assertIn("Недостаточно данных", question(data, "parallel")["reference"])
            self.assertIn(f"второй вывод {hidden} скрыт", question(data, "overall")["reference"])

    def test_history_idempotency_version_and_session_isolation(self):
        payload = {"message_id": "stable", "version": 0, "text": "вариант 4"}
        original = self.client.post(self.url + "/turns", json=payload)
        repeated = self.client.post(self.url + "/turns", json=payload)
        self.assertEqual(original.json(), repeated.json())
        stale = self.client.post(self.url + "/turns", json={**payload, "message_id": "different"})
        self.assertEqual(stale.status_code, 409)
        saved = self.client.get(self.url).json()
        self.assertEqual(len(saved["messages"]), 3)
        second = self.client.post("/api/sessions").json()
        self.assertEqual(second["state"]["phase"], "goal")
        self.assertNotEqual(second["id"], saved["id"])

    def test_model_match_needs_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            model = FixtureModel()
            with TestClient(create_app(Path(folder) / "test.sqlite3", model)) as client:
                session = client.post("/api/sessions").json()
                url = f"/api/sessions/{session['id']}/turns"
                match = client.post(url, json={"version": 0, "message_id": "free", "text": "Научимся читать схемы."}).json()
                self.assertEqual(match["state"]["phase"], "goal")
                self.assertTrue(match["state"]["pending"])
                confirmation = client.post(url, json={"version": 1, "message_id": "confirm", "text": "Подтвердить"}).json()
                self.assertEqual(confirmation["state"]["phase"], "trial")

    def test_incorrect_model_match_cannot_promote_unrelated_or_negative_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            model = FixtureModel()
            with TestClient(create_app(Path(folder) / "test.sqlite3", model)) as client:
                session = client.post("/api/sessions").json()
                url = f"/api/sessions/{session['id']}/turns"
                texts = (
                    "Кто такой Дональд Трамп",
                    "Будем готовить фруктовый салат.",
                    "Мы не будем определять вид соединения проводников, будем измерять температуру.",
                    "Схема.",
                    "Подтвердить",
                )
                for version, text in enumerate(texts):
                    result = client.post(url, json={
                        "version": version, "message_id": f"blocked-{version}", "text": text,
                    }).json()
                    self.assertFalse(result["state"]["pending"])
                    self.assertEqual(result["state"]["phase"], "goal")
                    if version == 0:
                        self.assertEqual(model.options, [])

    def test_knowledge_lookup_uses_approved_card(self):
        for text in ("Что такое узел?", "Что называется узлом"):
            result = self.send(text)
            self.assertIn("непрерывным проводом", result["message"]["text"])
            self.assertEqual(result["state"]["phase"], "goal")
