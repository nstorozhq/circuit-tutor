import json
import os
import sqlite3
from contextlib import asynccontextmanager, closing
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .accounts import Accounts, Credentials, check_origin
from .model import GigaChat
from .knowledge import lookup, norm
from .tutor import QUESTION_START, greeting, new_state, respond, verified_training_answer, view

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DIST = ROOT / "frontend/dist"


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message_id: str = Field(pattern=r"^[\w-]{1,100}$")
    version: int = Field(ge=0)
    text: str = Field(max_length=1200)


class Store:
    def __init__(self, path):
        self.path = path
        with closing(self.connect()) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, version INTEGER NOT NULL,
                    state TEXT NOT NULL, history TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS turns (
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    message_id TEXT NOT NULL, result TEXT NOT NULL,
                    PRIMARY KEY (session_id, message_id)
                );
                CREATE TABLE IF NOT EXISTS accounts (
                    id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE,
                    salt BLOB NOT NULL, password_hash BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS account_tokens (
                    digest BLOB NOT NULL PRIMARY KEY,
                    account_id INTEGER NOT NULL REFERENCES accounts(id),
                    expires_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lesson_owners (
                    session_id TEXT NOT NULL PRIMARY KEY REFERENCES sessions(id),
                    account_id INTEGER NOT NULL REFERENCES accounts(id)
                );
                CREATE INDEX IF NOT EXISTS lesson_owners_by_account ON lesson_owners(account_id);
                CREATE TABLE IF NOT EXISTS auth_limits (
                    key TEXT NOT NULL PRIMARY KEY,
                    expires_at INTEGER NOT NULL, attempts INTEGER NOT NULL
                );
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA foreign_keys = ON")
        db.row_factory = sqlite3.Row
        return db

    def create(self, account_id):
        session_id = uuid4().hex
        state = new_state()
        history = [{"id": "welcome", "role": "assistant", "text": greeting(state)}]
        with closing(self.connect()) as db, db:
            db.execute("INSERT INTO sessions VALUES (?, 0, ?, ?)",
                       (session_id, json.dumps(state), json.dumps(history)))
            if account_id is not None:
                db.execute("INSERT INTO lesson_owners VALUES (?, ?)", (session_id, account_id))
        return session_id

    def latest_lesson(self, account_id, guest_lesson=None):
        with closing(self.connect()) as db, db:
            if guest_lesson:
                db.execute("BEGIN IMMEDIATE")
            found = db.execute("""
                SELECT s.id FROM lesson_owners o JOIN sessions s ON s.id = o.session_id
                WHERE o.account_id = ? ORDER BY s.rowid DESC LIMIT 1
                """, (account_id,)).fetchone()
            if found:
                return found["id"]
            if guest_lesson:
                claimed = db.execute("""
                    INSERT OR IGNORE INTO lesson_owners
                    SELECT id, ? FROM sessions WHERE id = ?
                    """, (account_id, guest_lesson))
                if claimed.rowcount:
                    return guest_lesson
        return None

    @staticmethod
    def accessible_lesson(db, session_id, account_id):
        row = db.execute("""
            SELECT s.version, s.state, s.history
            FROM sessions s LEFT JOIN lesson_owners o ON o.session_id = s.id
            WHERE s.id = ? AND (o.account_id IS NULL OR o.account_id = ?)
            """, (session_id, account_id)).fetchone()
        if not row:
            raise HTTPException(404, "Урок не найден.")
        return row

    def load(self, session_id, account_id, message_id=None):
        with closing(self.connect()) as db:
            row = self.accessible_lesson(db, session_id, account_id)
            existing = self.previous(db, session_id, message_id) if message_id else None
        state = json.loads(row["state"])
        history = json.loads(row["history"])
        if state["phase"] == "goal":
            state.update(phase="trial", tries=0, pending=None)
            opening = {"id": "first-observation", "role": "assistant", "text": greeting(state)}
            if row["version"] == 0:
                history = [opening]
            else:
                history.append(opening)
        if state["phase"] == "trial" and any(
            item["role"] == "user" and verified_training_answer(state, item["text"]) for item in history
        ):
            state.update(phase="method", operation=0, pending=None, tries=0, case_correct=1)
            explanation = ""
            latest_user = next((item["text"] for item in reversed(history) if item["role"] == "user"), "")
            if QUESTION_START.match(norm(latest_user)):
                explanation = "\n\n".join(item["text"] for item in lookup(latest_user))
            message = "Ранее отправленный ответ о ветвях верный. Теперь составим способ решения."
            if explanation:
                message += "\n\n" + explanation
            history.append({
                "id": "recover-verified-answer", "role": "assistant",
                "text": message + "\n\n" + view(state)["question"],
            })
        if state["phase"] == "compare":
            state.update(phase="method", operation=0, pending=None, tries=0)
            history.append({
                "id": "continue-with-nodes", "role": "assistant",
                "text": "Составим порядок разбора схемы. " + view(state)["question"],
            })
        return row["version"], state, history, existing

    @staticmethod
    def previous(db, session_id, message_id):
        row = db.execute("SELECT result FROM turns WHERE session_id = ? AND message_id = ?",
                         (session_id, message_id)).fetchone()
        return json.loads(row["result"]) if row else None

    def commit(self, session_id, account_id, turn, state, history, result):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            self.accessible_lesson(db, session_id, account_id)
            existing = self.previous(db, session_id, turn.message_id)
            if existing:
                return existing
            changed = db.execute(
                "UPDATE sessions SET version = version + 1, state = ?, history = ? WHERE id = ? AND version = ?",
                (json.dumps(state), json.dumps(history), session_id, turn.version)
            )
            if not changed.rowcount:
                raise HTTPException(409, "Урок уже обновился. Обнови страницу.")
            db.execute("INSERT INTO turns VALUES (?, ?, ?)",
                       (session_id, turn.message_id, json.dumps(result)))
            return result


def create_app(db_path=None, model=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.store = Store(db_path or os.getenv("TUTOR_DB", str(ROOT / "tutor.sqlite3")))
        app.state.accounts = Accounts(app.state.store.connect)
        app.state.model = model if model is not None else GigaChat()
        yield

    app = FastAPI(title="Цепь · учебный помощник", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def prevent_private_caching(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request: Request, error: RequestValidationError):
        if request.url.path.startswith("/api/auth/"):
            return JSONResponse({"detail": "Проверь логин и пароль."}, status_code=422)
        return await request_validation_exception_handler(request, error)

    def profile(user, guest_lesson=None):
        if user is None:
            return {"user": None, "lesson_id": None}
        return {
            "user": user["username"],
            "lesson_id": app.state.store.latest_lesson(user["id"], guest_lesson),
        }

    def start_account(request, response, credentials, register):
        check_origin(request)
        user = app.state.accounts.authenticate(request, credentials, register)
        result = profile(user, credentials.guest_lesson)
        app.state.accounts.start(request, response, user["id"])
        return result

    def account_id(request):
        user = app.state.accounts.current(request)
        return user["id"] if user else None

    @app.get("/api/health")
    def health():
        return {"ok": True, "gigachat_configured": app.state.model.enabled}

    @app.get("/api/auth")
    def current_account(request: Request):
        return profile(app.state.accounts.current(request))

    @app.post("/api/auth/register", status_code=201)
    def register(request: Request, response: Response, credentials: Credentials):
        return start_account(request, response, credentials, register=True)

    @app.post("/api/auth/login")
    def login(request: Request, response: Response, credentials: Credentials):
        return start_account(request, response, credentials, register=False)

    @app.post("/api/auth/logout")
    def logout(request: Request, response: Response):
        check_origin(request)
        app.state.accounts.end(request, response)
        return profile(None)

    def lesson_response(session_id, user_id):
        version, state, history, _ = app.state.store.load(session_id, user_id)
        return {"id": session_id, "version": version, "state": view(state), "messages": history,
                "gigachat_configured": app.state.model.enabled}

    @app.post("/api/sessions", status_code=201)
    def create_session(request: Request):
        check_origin(request)
        user_id = account_id(request)
        return lesson_response(app.state.store.create(user_id), user_id)

    @app.get("/api/sessions/{session_id}")
    def read_session(session_id: str, request: Request):
        return lesson_response(session_id, account_id(request))

    @app.post("/api/sessions/{session_id}/turns")
    async def send(session_id: str, turn: Turn, request: Request):
        check_origin(request)
        user_id = account_id(request)
        version, state, history, existing = app.state.store.load(session_id, user_id, turn.message_id)
        if existing:
            return existing
        if version != turn.version:
            raise HTTPException(409, "Урок уже обновился. Обнови страницу.")
        if version >= 300:
            raise HTTPException(409, "Достигнут лимит урока. Начни новую сессию.")
        reply = await respond(state, turn.text, app.state.model, history=history[-6:])
        message = {"id": "reply-" + turn.message_id, "role": "assistant", "text": reply}
        grounding = state.pop("last_rag", None)
        if grounding:
            message["grounding"] = grounding
        history.extend([{"id": turn.message_id, "role": "user", "text": turn.text}, message])
        result = {"version": version + 1, "state": view(state), "message": message}
        return app.state.store.commit(session_id, user_id, turn, state, history, result)

    if (DIST / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        if not (DIST / "index.html").is_file():
            raise HTTPException(503, "Сначала соберите фронтенд командой pnpm build или запустите pnpm dev.")
        return FileResponse(DIST / "index.html")

    return app


app = create_app()
