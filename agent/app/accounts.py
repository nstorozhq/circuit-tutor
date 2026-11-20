import hashlib
import hmac
import secrets
import sqlite3
import time
from contextlib import closing

from fastapi import HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, SecretStr

SESSION_AGE = 14 * 24 * 60 * 60


class Credentials(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(pattern=r"^[a-zA-Z0-9_.-]{3,32}$")
    password: SecretStr
    guest_lesson: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


def check_origin(request: Request):
    origin = request.headers.get("origin")
    if origin is not None and origin != f"{request.url.scheme}://{request.url.netloc}":
        raise HTTPException(403, "Запрос отклонён.")


def session_cookie(request: Request):
    return "__Host-circuit_account" if request.url.scheme == "https" else "circuit_account"


def password_hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)


def token_hash(token):
    return hashlib.sha256(token.encode()).digest()


class Accounts:
    def __init__(self, connect):
        self.connect = connect

    def check_rate(self, request, username):
        now = int(time.time())
        ip = request.client.host if request.client else "unknown"
        buckets = ((f"ip:{ip}", 60, 20), (f"user:{username}", 300, 10), ("global", 60, 120))
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM auth_limits WHERE expires_at <= ?", (now,))
            for key, width, limit in buckets:
                expires_at = (now // width + 1) * width
                found = db.execute("SELECT attempts FROM auth_limits WHERE key = ?", (key,)).fetchone()
                if found and found["attempts"] >= limit:
                    raise HTTPException(429, "Слишком много попыток. Попробуй позже.",
                                        headers={"Retry-After": str(expires_at - now)})
                db.execute("""
                    INSERT INTO auth_limits VALUES (?, ?, 1)
                    ON CONFLICT(key) DO UPDATE SET attempts = attempts + 1
                    """, (key, expires_at))

    def authenticate(self, request, credentials, register):
        username = credentials.username.lower()
        password = credentials.password.get_secret_value()
        if not 12 <= len(password) <= 128:
            raise HTTPException(400, "Пароль должен содержать от 12 до 128 символов.")
        self.check_rate(request, username)

        with closing(self.connect()) as db:
            if register:
                salt = secrets.token_bytes(16)
                digest = password_hash(password, salt)
                try:
                    with db:
                        cursor = db.execute("INSERT INTO accounts (username, salt, password_hash) VALUES (?, ?, ?)",
                                            (username, salt, digest))
                except sqlite3.IntegrityError:
                    raise HTTPException(409, "Этот логин уже занят.") from None
                return {"id": cursor.lastrowid, "username": username}

            row = db.execute("SELECT * FROM accounts WHERE username = ?", (username,)).fetchone()
        digest = password_hash(password, row["salt"] if row else b"\x00" * 16)
        if not hmac.compare_digest(digest, row["password_hash"] if row else b"\x00" * 32) or not row:
            raise HTTPException(401, "Неверный логин или пароль.")
        return {"id": row["id"], "username": row["username"]}

    def current(self, request: Request):
        token = request.cookies.get(session_cookie(request))
        if not token:
            return None
        with closing(self.connect()) as db:
            row = db.execute("""
                SELECT a.id, a.username FROM account_tokens t
                JOIN accounts a ON a.id = t.account_id
                WHERE t.digest = ? AND t.expires_at > ?
                """, (token_hash(token), int(time.time()))).fetchone()
        return dict(row) if row else None

    def start(self, request: Request, response: Response, account_id: int):
        cookie = session_cookie(request)
        token = secrets.token_urlsafe(32)
        previous = request.cookies.get(cookie)
        now = int(time.time())
        with closing(self.connect()) as db, db:
            db.execute("DELETE FROM account_tokens WHERE expires_at <= ?", (now,))
            if previous:
                db.execute("DELETE FROM account_tokens WHERE digest = ?", (token_hash(previous),))
            db.execute("INSERT INTO account_tokens VALUES (?, ?, ?)", (token_hash(token), account_id, now + SESSION_AGE))
        response.set_cookie(cookie, token, max_age=SESSION_AGE, httponly=True,
                            secure=request.url.scheme == "https", samesite="strict")

    def end(self, request: Request, response: Response):
        cookie = session_cookie(request)
        token = request.cookies.get(cookie)
        if token:
            with closing(self.connect()) as db, db:
                db.execute("DELETE FROM account_tokens WHERE digest = ?", (token_hash(token),))
        response.delete_cookie(cookie, httponly=True, secure=request.url.scheme == "https", samesite="strict")
