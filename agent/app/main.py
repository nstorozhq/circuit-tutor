import json
import os
import sqlite3
from contextlib import asynccontextmanager, closing
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .model import GigaChat
from .tutor import greeting, new_state, respond, view

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
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA foreign_keys = ON")
        db.row_factory = sqlite3.Row
        return db

    def create(self):
        session_id = uuid4().hex
        state = new_state()
        history = [{"id": "welcome", "role": "assistant", "text": greeting(state)}]
        with closing(self.connect()) as db, db:
            db.execute("INSERT INTO sessions VALUES (?, 0, ?, ?)",
                       (session_id, json.dumps(state), json.dumps(history)))
        return session_id

    def load(self, session_id, message_id=None):
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if not row:
                raise HTTPException(404, "Урок не найден.")
            existing = self.previous(db, session_id, message_id) if message_id else None
            return row["version"], json.loads(row["state"]), json.loads(row["history"]), existing

    @staticmethod
    def previous(db, session_id, message_id):
        row = db.execute("SELECT result FROM turns WHERE session_id = ? AND message_id = ?",
                         (session_id, message_id)).fetchone()
        return json.loads(row["result"]) if row else None

    def commit(self, session_id, turn, state, history, result):
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
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
        app.state.model = model if model is not None else GigaChat()
        yield

    app = FastAPI(title="Цепь · учебный помощник", version="1.0.0", lifespan=lifespan)

    @app.get("/api/health")
    def health():
        return {"ok": True, "gigachat_configured": app.state.model.enabled}

    @app.post("/api/sessions", status_code=201)
    def create_session():
        return read_session(app.state.store.create())

    @app.get("/api/sessions/{session_id}")
    def read_session(session_id: str):
        version, state, history, _ = app.state.store.load(session_id)
        return {"id": session_id, "version": version, "state": view(state), "messages": history,
                "gigachat_configured": app.state.model.enabled}

    @app.post("/api/sessions/{session_id}/turns")
    async def send(session_id: str, turn: Turn):
        version, state, history, existing = app.state.store.load(session_id, turn.message_id)
        if existing:
            return existing
        if version != turn.version:
            raise HTTPException(409, "Урок уже обновился. Обнови страницу.")
        if version >= 300:
            raise HTTPException(409, "Достигнут лимит урока. Начни новую сессию.")
        reply = await respond(state, turn.text, app.state.model)
        message = {"id": "reply-" + turn.message_id, "role": "assistant", "text": reply}
        history.extend([{"id": turn.message_id, "role": "user", "text": turn.text}, message])
        result = {"version": version + 1, "state": view(state), "message": message}
        return app.state.store.commit(session_id, turn, state, history, result)

    if (DIST / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        if not (DIST / "index.html").is_file():
            raise HTTPException(503, "Сначала соберите фронтенд командой pnpm build или запустите pnpm dev.")
        return FileResponse(DIST / "index.html")

    return app


app = create_app()
