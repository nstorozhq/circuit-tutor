import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.knowledge import CASES, INTRO, question
from app.rag import RagResult, retrieve, validate_result


class RagModel:
    enabled = True

    def __init__(self):
        self.requests = []
        self.status = "matched"
        self.payload = {
            "intent": "answer", "option": None, "evidence": "", "quotes": [],
            "branches": [
                {"position": "upper", "elements": ["R1"], "evidence": "r1 на верхней"},
                {"position": "upper", "elements": ["R2"], "evidence": "r2 на верхней"},
                {"position": "lower", "elements": ["R3"], "evidence": "r3 на нижней"},
            ],
        }

    async def rag(self, text, task, sources, *, history=()):
        self.requests.append((text, task, sources))
        if self.status != "matched":
            return None, self.status
        return validate_result(RagResult.model_validate(self.payload), text, sources, task["choices"]), "matched"


class RagTests(unittest.TestCase):
    def test_uncertainty_explains_named_components_without_grading(self):
        model = RagModel()
        with tempfile.TemporaryDirectory() as folder:
            with TestClient(create_app(Path(folder) / "test.db", model)) as client:
                session = client.post("/api/sessions").json()
                path = f"/api/sessions/{session['id']}/turns"
                session = client.post(path, json={
                    "version": 0, "message_id": "start",
                    "text": "r1 на верхней r2 на верхней\nr3 на нижней",
                }).json()
                model.payload = {"intent": "unsure", "branches": [], "quotes": []}
                for status in ("matched", "unclear", "unavailable"):
                    model.status = status
                    session = client.post(path, json={
                        "version": session["version"], "message_id": status,
                        "text": "не уверен на счет связи р1 и р2",
                    }).json()
                    self.assertEqual(session["state"]["step_label"], "Узлы")
                    self.assertFalse(session["state"]["pending"])
                    self.assertIn("R1 подключён между A и C", session["message"]["text"])
                    self.assertIn("R2 подключён между C и B", session["message"]["text"])
                    self.assertIn("Общие узлы R1 и R2 — C", session["message"]["text"])
                    self.assertNotIn("Не удалось проверить", session["message"]["text"])
                    sources = model.requests[-1][2]
                    self.assertIn("C01-component-help", [s["id"] for s in sources])
                    self.assertNotIn("C01-nodes", [s["id"] for s in sources])

    def test_free_answer_uses_retrieval_and_merges_repeated_branches(self):
        model = RagModel()
        with tempfile.TemporaryDirectory() as folder:
            with TestClient(create_app(Path(folder) / "test.db", model)) as client:
                session = client.post("/api/sessions").json()
                path = f"/api/sessions/{session['id']}/turns"
                text = "r1 на верхней r2 на верхней\nr3 на нижней"
                result = client.post(path, json={"version": 0, "message_id": "rag", "text": text}).json()
                self.assertEqual(result["state"]["step_label"], "Узлы")
                self.assertIn("Верно", result["message"]["text"])
                self.assertEqual(model.requests[0][0], text)
                sources = model.requests[0][2]
                self.assertTrue(any(source["id"] == "C01-drawing" for source in sources))
                self.assertFalse(any("C09" in source["id"] for source in sources))
                model.payload["branches"].pop(1)
                another = client.post("/api/sessions").json()
                result = client.post(f"/api/sessions/{another['id']}/turns",
                                     json={"version": 0, "message_id": "partial", "text": "r1 на верхней\nr3 на нижней"}).json()
                self.assertEqual(result["state"]["phase"], "trial")

    def test_ungrounded_quotes_and_invented_elements_are_rejected(self):
        sources = [{"id": "concept-0", "text": "Узел объединяет точки."}]
        base = {"intent": "question", "option": None, "evidence": "", "branches": [], "quotes": []}
        for quote in ({"source_id": "unknown", "text": "Узел объединяет точки."},
                      {"source_id": "concept-0", "text": "Париж — столица Франции."}):
            result = RagResult.model_validate({**base, "quotes": [quote]})
            with self.assertRaises(ValueError):
                validate_result(result, "Что такое узел?", sources, [])
        result = RagResult.model_validate({**base, "intent": "answer", "branches": [
            {"position": "upper", "elements": ["R2"], "evidence": "R1 сверху"},
        ]})
        with self.assertRaises(ValueError):
            validate_result(result, "R1 сверху", sources, [])

    def test_hint_context_has_no_reference_and_control_has_no_sources(self):
        task = question(CASES["C01"], "series")
        hints = retrieve(CASES["C01"], task, "Подскажи, как найти последовательные резисторы")
        self.assertFalse(any(source["id"] == "C01-series" for source in hints))
        self.assertTrue(any(source["id"] == "C01-series-hint" for source in hints))
        self.assertEqual(retrieve(CASES["C09"], task, "Помоги", control=True), [])
        sources = retrieve(CASES["C01"], INTRO["trial"], "Что такое узел?")
        self.assertIn("concept-0", [source["id"] for source in sources])
