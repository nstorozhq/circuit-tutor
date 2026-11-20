import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import create_app


class AccountsTests(unittest.TestCase):
    credentials = {"username": "student_test", "password": "test-password-254321"}

    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.db = Path(temporary) / "lessons.sqlite3"
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("""
                CREATE TABLE sessions (
                    id TEXT PRIMARY KEY, version INTEGER NOT NULL,
                    state TEXT NOT NULL, history TEXT NOT NULL
                )
                """)
        self.app = create_app(self.db)
        self.browser = self.enterContext(TestClient(self.app, base_url="https://lesson.test"))

    def another_browser(self):
        return TestClient(self.app, base_url="https://lesson.test")

    def test_claim_existing_guest_lesson_and_resume_after_sign_in(self):
        guest = self.browser.post("/api/sessions").json()
        path = f"/api/sessions/{guest['id']}"
        turn = {"version": 0, "message_id": "start", "text": "R1 и R2 сверху, R3 снизу."}
        self.assertEqual(self.browser.post(path + "/turns", json=turn).json()["state"]["phase"], "guided")

        registered = self.browser.post("/api/auth/register", json={**self.credentials, "guest_lesson": guest["id"]})
        self.assertEqual(registered.status_code, 201)
        self.assertEqual(registered.json(), {"user": "student_test", "lesson_id": guest["id"]})
        cookie = registered.headers["set-cookie"]
        self.assertIn("__Host-circuit_account=", cookie)
        self.assertIn("Secure", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=strict", cookie)
        self.assertEqual(self.browser.get(path).json()["version"], 1)

        outsider = self.another_browser()
        self.assertEqual(outsider.get(path).status_code, 404)
        self.assertEqual(outsider.post(path + "/turns", json=turn).status_code, 404)
        attempted_claim = outsider.post("/api/auth/register", json={
            **self.credentials, "username": "other_student", "guest_lesson": guest["id"],
        })
        self.assertEqual(attempted_claim.status_code, 201)
        self.assertIsNone(attempted_claim.json()["lesson_id"])
        self.assertEqual(outsider.get(path).status_code, 404)

        logged_out = self.browser.post("/api/auth/logout")
        self.assertIsNone(logged_out.json()["user"])
        prior_cookie = cookie.split(";", 1)[0]
        self.assertEqual(self.browser.get(path, headers={"Cookie": prior_cookie}).status_code, 404)

        fresh_browser = self.another_browser()
        login = fresh_browser.post("/api/auth/login", json=self.credentials)
        self.assertEqual(login.json()["lesson_id"], guest["id"])
        self.assertEqual(fresh_browser.get(path).json()["state"]["phase"], "guided")
        new_lesson = fresh_browser.post("/api/sessions").json()
        self.assertNotEqual(new_lesson["id"], guest["id"])
        self.assertEqual(fresh_browser.get("/api/auth").json()["lesson_id"], new_lesson["id"])
        self.assertEqual(outsider.get(f"/api/sessions/{new_lesson['id']}").status_code, 404)

    def test_validation_and_wrong_origin(self):
        response = self.browser.post("/api/auth/register", json=self.credentials,
                                     headers={"Origin": "https://elsewhere.test"})
        self.assertEqual(response.status_code, 403)
        bad_form = self.browser.post("/api/auth/register", json={**self.credentials, "username": "?"})
        self.assertEqual(bad_form.status_code, 422)
        self.assertNotIn(self.credentials["password"], bad_form.text)
        unknown = self.browser.post("/api/auth/login", json=self.credentials)
        self.assertEqual(unknown.status_code, 401)
        self.assertEqual(unknown.headers["cache-control"], "no-store")

    def test_repeated_auth_requests_are_limited(self):
        with patch("app.accounts.time.time", return_value=1_800_000_000):
            with patch("app.accounts.password_hash", return_value=b"x" * 32):
                for _ in range(10):
                    self.assertEqual(self.browser.post("/api/auth/login", json=self.credentials).status_code, 401)
                throttled = self.browser.post("/api/auth/login", json=self.credentials)
        self.assertEqual(throttled.status_code, 429)
        self.assertEqual(throttled.headers["retry-after"], "300")
