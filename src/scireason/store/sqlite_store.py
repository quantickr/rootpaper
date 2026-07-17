# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""SQLite-реализация общего хранилища (офлайн-режим по умолчанию).

Один файл, стандартная библиотека, потокобезопасность за счёт
короткоживущих соединений + ``threading.Lock`` на запись и WAL-режима.
"""

import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional

from .base import (
    ALL_SOURCES,
    ROLE_USER,
    EmailToken,
    ReportRow,
    Store,
    User,
    UserSettings,
)

DEFAULT_DB_PATH = Path("runs") / "tg_bot.sqlite3"


class SqliteStore(Store):
    """Хранилище на SQLite."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    email          TEXT    UNIQUE,
                    password_hash  TEXT,
                    role           TEXT    NOT NULL DEFAULT 'user',
                    email_verified INTEGER NOT NULL DEFAULT 0,
                    tg_user_id     INTEGER UNIQUE,
                    tg_username    TEXT,
                    display_name   TEXT,
                    created_at     REAL    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS email_tokens (
                    token      TEXT    PRIMARY KEY,
                    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    purpose    TEXT    NOT NULL,
                    created_at REAL    NOT NULL,
                    expires_at REAL    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_settings (
                    owner_id     INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                    sources      TEXT    NOT NULL,
                    search_limit INTEGER NOT NULL,
                    top_papers   INTEGER NOT NULL,
                    updated_at   REAL    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reports (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id   INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    query      TEXT    NOT NULL,
                    html       BLOB    NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    origin     TEXT    NOT NULL DEFAULT 'web',
                    created_at REAL    NOT NULL
                );

                CREATE TABLE IF NOT EXISTS report_shares (
                    token      TEXT    PRIMARY KEY,
                    report_id  INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
                    created_at REAL    NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_reports_owner
                    ON reports (owner_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_tokens_user
                    ON email_tokens (user_id);
                CREATE INDEX IF NOT EXISTS idx_shares_report
                    ON report_shares (report_id);
                """
            )

    # ----------------------------------------------------------------- users
    def _row_to_user(self, row: sqlite3.Row) -> User:
        return User(
            id=int(row["id"]),
            email=row["email"],
            password_hash=row["password_hash"],
            role=str(row["role"]),
            email_verified=bool(row["email_verified"]),
            tg_user_id=int(row["tg_user_id"]) if row["tg_user_id"] is not None else None,
            tg_username=row["tg_username"],
            display_name=row["display_name"],
            created_at=float(row["created_at"]),
        )

    def create_user(
        self,
        *,
        email: Optional[str] = None,
        password_hash: Optional[str] = None,
        role: str = ROLE_USER,
        email_verified: bool = False,
        tg_user_id: Optional[int] = None,
        tg_username: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> User:
        now = time.time()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO users
                    (email, password_hash, role, email_verified,
                     tg_user_id, tg_username, display_name, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    email,
                    password_hash,
                    role,
                    1 if email_verified else 0,
                    tg_user_id,
                    tg_username,
                    display_name,
                    now,
                ),
            )
            uid = int(cur.lastrowid)
        return User(
            id=uid,
            email=email,
            password_hash=password_hash,
            role=role,
            email_verified=email_verified,
            tg_user_id=tg_user_id,
            tg_username=tg_username,
            display_name=display_name,
            created_at=now,
        )

    def get_user(self, user_id: int) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return self._row_to_user(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email.lower().strip(),)
            ).fetchone()
        return self._row_to_user(row) if row else None

    def get_user_by_tg(self, tg_user_id: int) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE tg_user_id = ?", (int(tg_user_id),)
            ).fetchone()
        return self._row_to_user(row) if row else None

    def update_user(self, user: User) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE users SET
                    email = ?, password_hash = ?, role = ?, email_verified = ?,
                    tg_user_id = ?, tg_username = ?, display_name = ?
                WHERE id = ?
                """,
                (
                    user.email,
                    user.password_hash,
                    user.role,
                    1 if user.email_verified else 0,
                    user.tg_user_id,
                    user.tg_username,
                    user.display_name,
                    user.id,
                ),
            )

    def get_or_create_tg_user(
        self, tg_user_id: int, *, tg_username: Optional[str] = None
    ) -> User:
        existing = self.get_user_by_tg(tg_user_id)
        if existing is not None:
            if tg_username and existing.tg_username != tg_username:
                existing.tg_username = tg_username
                self.update_user(existing)
            return existing
        return self.create_user(
            tg_user_id=tg_user_id,
            tg_username=tg_username,
            email_verified=False,
        )

    def link_tg_to_user(
        self, user_id: int, tg_user_id: int, *, tg_username: Optional[str] = None
    ) -> None:
        with self._lock, self._connect() as conn:
            shadow = conn.execute(
                "SELECT id FROM users WHERE tg_user_id = ?", (int(tg_user_id),)
            ).fetchone()
            if shadow is not None and int(shadow["id"]) != user_id:
                shadow_id = int(shadow["id"])
                # Переносим отчёты теневого пользователя, если у целевого их нет
                # для той же записи (просто перепривязываем owner_id).
                conn.execute(
                    "UPDATE reports SET owner_id = ? WHERE owner_id = ?",
                    (user_id, shadow_id),
                )
                # Настройки теневого не переносим: у веб-аккаунта свои.
                conn.execute("DELETE FROM users WHERE id = ?", (shadow_id,))
            conn.execute(
                "UPDATE users SET tg_user_id = ?, tg_username = ? WHERE id = ?",
                (int(tg_user_id), tg_username, user_id),
            )

    def count_users(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()
        return int(row["n"]) if row else 0

    # ------------------------------------------------------- email tokens
    def create_email_token(
        self, user_id: int, *, purpose: str, ttl_seconds: int
    ) -> EmailToken:
        token = secrets.token_urlsafe(32)
        now = time.time()
        exp = now + ttl_seconds
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO email_tokens (token, user_id, purpose, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (token, user_id, purpose, now, exp),
            )
        return EmailToken(
            token=token,
            user_id=user_id,
            purpose=purpose,
            created_at=now,
            expires_at=exp,
        )

    def consume_email_token(self, token: str, *, purpose: str) -> Optional[int]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT user_id, purpose, expires_at FROM email_tokens WHERE token = ?",
                (token,),
            ).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM email_tokens WHERE token = ?", (token,))
            if str(row["purpose"]) != purpose:
                return None
            if float(row["expires_at"]) < time.time():
                return None
            return int(row["user_id"])

    def create_email_code(
        self, user_id: int, *, purpose: str, ttl_seconds: int
    ) -> EmailToken:
        now = time.time()
        exp = now + ttl_seconds
        with self._lock, self._connect() as conn:
            # Оставляем один активный код на пользователя+назначение.
            conn.execute(
                "DELETE FROM email_tokens WHERE user_id = ? AND purpose = ?",
                (user_id, purpose),
            )
            # token — PRIMARY KEY, поэтому 6-значный код должен быть уникальным
            # среди всех активных токенов; при коллизии генерируем заново.
            for _ in range(50):
                code = f"{secrets.randbelow(1_000_000):06d}"
                try:
                    conn.execute(
                        """
                        INSERT INTO email_tokens
                            (token, user_id, purpose, created_at, expires_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (code, user_id, purpose, now, exp),
                    )
                    break
                except sqlite3.IntegrityError:
                    continue
            else:  # pragma: no cover - крайне маловероятно
                raise RuntimeError("Не удалось сгенерировать уникальный код")
        return EmailToken(
            token=code,
            user_id=user_id,
            purpose=purpose,
            created_at=now,
            expires_at=exp,
        )

    def consume_email_code(
        self, user_id: int, code: str, *, purpose: str
    ) -> bool:
        code = (code or "").strip()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT expires_at FROM email_tokens "
                "WHERE user_id = ? AND token = ? AND purpose = ?",
                (user_id, code, purpose),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                "DELETE FROM email_tokens WHERE user_id = ? AND token = ? AND purpose = ?",
                (user_id, code, purpose),
            )
            if float(row["expires_at"]) < time.time():
                return False
            return True

    # ----------------------------------------------------------------- settings
    def get_settings(self, owner_id: int) -> UserSettings:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT sources, search_limit, top_papers "
                "FROM user_settings WHERE owner_id = ?",
                (owner_id,),
            ).fetchone()
        if row is None:
            return UserSettings()
        try:
            sources = json.loads(row["sources"])
            if not isinstance(sources, list):
                sources = list(ALL_SOURCES)
        except (json.JSONDecodeError, TypeError):
            sources = list(ALL_SOURCES)
        sources = [s for s in sources if s in ALL_SOURCES]
        return UserSettings(
            sources=sources,
            search_limit=int(row["search_limit"]),
            top_papers=int(row["top_papers"]),
        )

    def save_settings(self, owner_id: int, s: UserSettings) -> None:
        payload = json.dumps(list(s.sources))
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_settings
                    (owner_id, sources, search_limit, top_papers, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(owner_id) DO UPDATE SET
                    sources      = excluded.sources,
                    search_limit = excluded.search_limit,
                    top_papers   = excluded.top_papers,
                    updated_at   = excluded.updated_at
                """,
                (owner_id, payload, int(s.search_limit), int(s.top_papers), time.time()),
            )

    # ----------------------------------------------------------------- reports
    def add_report(
        self, owner_id: int, query: str, html: bytes, *, origin: str = "web"
    ) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO reports (owner_id, query, html, size_bytes, origin, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (owner_id, query, html, len(html), origin, time.time()),
            )
            return int(cur.lastrowid)

    def count_reports(self, owner_id: int) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM reports WHERE owner_id = ?",
                (owner_id,),
            ).fetchone()
        return int(row["n"]) if row else 0

    def list_reports(
        self, owner_id: int, *, offset: int, limit: int
    ) -> List[ReportRow]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, query, created_at, size_bytes, origin
                FROM reports
                WHERE owner_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (owner_id, int(limit), int(offset)),
            ).fetchall()
        return [
            ReportRow(
                id=int(r["id"]),
                query=str(r["query"]),
                created_at=float(r["created_at"]),
                size_bytes=int(r["size_bytes"]),
                origin=str(r["origin"]),
            )
            for r in rows
        ]

    def get_report_html(self, owner_id: int, report_id: int) -> Optional[bytes]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT html FROM reports WHERE id = ? AND owner_id = ?",
                (int(report_id), owner_id),
            ).fetchone()
        if row is None:
            return None
        return bytes(row["html"])

    # ------------------------------------------------------------ report shares
    def create_report_share(self, report_id: int) -> str:
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT token FROM report_shares WHERE report_id = ?",
                (int(report_id),),
            ).fetchone()
            if existing is not None:
                return str(existing["token"])
            for _ in range(50):
                token = secrets.token_urlsafe(24)
                try:
                    conn.execute(
                        "INSERT INTO report_shares (token, report_id, created_at) "
                        "VALUES (?, ?, ?)",
                        (token, int(report_id), time.time()),
                    )
                    return token
                except sqlite3.IntegrityError:
                    continue
            raise RuntimeError("Не удалось сгенерировать уникальный share-токен")

    def get_report_html_by_share(self, token: str) -> Optional[bytes]:
        token = (token or "").strip()
        if not token:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT r.html AS html FROM report_shares s "
                "JOIN reports r ON r.id = s.report_id "
                "WHERE s.token = ?",
                (token,),
            ).fetchone()
        if row is None:
            return None
        return bytes(row["html"])

    # ------------------------------------------------------------------- stats
    def stats_global(self) -> dict:
        with self._connect() as conn:
            users = int(conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"])
            reports = int(
                conn.execute("SELECT COUNT(*) AS n FROM reports").fetchone()["n"]
            )
            total_bytes_row = conn.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) AS s FROM reports"
            ).fetchone()
            total_bytes = int(total_bytes_row["s"])
            verified = int(
                conn.execute(
                    "SELECT COUNT(*) AS n FROM users WHERE email_verified = 1"
                ).fetchone()["n"]
            )
            tg_linked = int(
                conn.execute(
                    "SELECT COUNT(*) AS n FROM users WHERE tg_user_id IS NOT NULL"
                ).fetchone()["n"]
            )
            top_rows = conn.execute(
                """
                SELECT query, COUNT(*) AS n FROM reports
                GROUP BY query ORDER BY n DESC LIMIT 10
                """
            ).fetchall()
            by_origin = conn.execute(
                "SELECT origin, COUNT(*) AS n FROM reports GROUP BY origin"
            ).fetchall()
        return {
            "users": users,
            "verified_users": verified,
            "tg_linked_users": tg_linked,
            "reports": reports,
            "total_bytes": total_bytes,
            "top_queries": [(str(r["query"]), int(r["n"])) for r in top_rows],
            "by_origin": {str(r["origin"]): int(r["n"]) for r in by_origin},
        }

    def stats_for_owner(self, owner_id: int) -> dict:
        with self._connect() as conn:
            reports = int(
                conn.execute(
                    "SELECT COUNT(*) AS n FROM reports WHERE owner_id = ?",
                    (owner_id,),
                ).fetchone()["n"]
            )
            total_bytes = int(
                conn.execute(
                    "SELECT COALESCE(SUM(size_bytes), 0) AS s FROM reports WHERE owner_id = ?",
                    (owner_id,),
                ).fetchone()["s"]
            )
            top_rows = conn.execute(
                """
                SELECT query, COUNT(*) AS n FROM reports
                WHERE owner_id = ?
                GROUP BY query ORDER BY n DESC LIMIT 10
                """,
                (owner_id,),
            ).fetchall()
        return {
            "reports": reports,
            "total_bytes": total_bytes,
            "top_queries": [(str(r["query"]), int(r["n"])) for r in top_rows],
        }
