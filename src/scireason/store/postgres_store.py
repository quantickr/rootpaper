# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""PostgreSQL-реализация общего хранилища.

Используется, когда задан ``DATABASE_URL`` (см. :func:`store.factory.get_store`).
Надёжнее SQLite при конкурентной записи из двух процессов (бот + сайт).

Требует пакет ``psycopg[binary]`` (extra ``web``). Пул соединений —
``psycopg_pool.ConnectionPool``.
"""

import json
import secrets
import time
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

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id             BIGSERIAL PRIMARY KEY,
    email          TEXT UNIQUE,
    password_hash  TEXT,
    role           TEXT NOT NULL DEFAULT 'user',
    email_verified BOOLEAN NOT NULL DEFAULT FALSE,
    tg_user_id     BIGINT UNIQUE,
    tg_username    TEXT,
    display_name   TEXT,
    created_at     DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS email_tokens (
    token      TEXT PRIMARY KEY,
    user_id    BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    purpose    TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS user_settings (
    owner_id     BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    sources      TEXT NOT NULL,
    search_limit INTEGER NOT NULL,
    top_papers   INTEGER NOT NULL,
    updated_at   DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id         BIGSERIAL PRIMARY KEY,
    owner_id   BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    query      TEXT NOT NULL,
    html       BYTEA NOT NULL,
    size_bytes INTEGER NOT NULL,
    origin     TEXT NOT NULL DEFAULT 'web',
    created_at DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reports_owner ON reports (owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_tokens_user ON email_tokens (user_id);
"""


class PostgresStore(Store):
    """Хранилище на PostgreSQL (psycopg 3 + connection pool)."""

    def __init__(self, dsn: str) -> None:
        try:
            from psycopg_pool import ConnectionPool
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "PostgreSQL backend requires psycopg. Install with: "
                "pip install -e '.[web]'"
            ) from e
        # open=True: пул подключается сразу; min_size держит тёплые соединения.
        self._pool = ConnectionPool(dsn, min_size=1, max_size=10, open=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._pool.connection() as conn:
            conn.execute(_SCHEMA)

    # ----------------------------------------------------------------- users
    @staticmethod
    def _row_to_user(row) -> User:
        return User(
            id=int(row[0]),
            email=row[1],
            password_hash=row[2],
            role=str(row[3]),
            email_verified=bool(row[4]),
            tg_user_id=int(row[5]) if row[5] is not None else None,
            tg_username=row[6],
            display_name=row[7],
            created_at=float(row[8]),
        )

    _USER_COLS = (
        "id, email, password_hash, role, email_verified, "
        "tg_user_id, tg_username, display_name, created_at"
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
        with self._pool.connection() as conn:
            row = conn.execute(
                f"""
                INSERT INTO users
                    (email, password_hash, role, email_verified,
                     tg_user_id, tg_username, display_name, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING {self._USER_COLS}
                """,
                (
                    email,
                    password_hash,
                    role,
                    email_verified,
                    tg_user_id,
                    tg_username,
                    display_name,
                    now,
                ),
            ).fetchone()
        return self._row_to_user(row)

    def get_user(self, user_id: int) -> Optional[User]:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT {self._USER_COLS} FROM users WHERE id = %s", (user_id,)
            ).fetchone()
        return self._row_to_user(row) if row else None

    def get_user_by_email(self, email: str) -> Optional[User]:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT {self._USER_COLS} FROM users WHERE email = %s",
                (email.lower().strip(),),
            ).fetchone()
        return self._row_to_user(row) if row else None

    def get_user_by_tg(self, tg_user_id: int) -> Optional[User]:
        with self._pool.connection() as conn:
            row = conn.execute(
                f"SELECT {self._USER_COLS} FROM users WHERE tg_user_id = %s",
                (int(tg_user_id),),
            ).fetchone()
        return self._row_to_user(row) if row else None

    def update_user(self, user: User) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                """
                UPDATE users SET
                    email = %s, password_hash = %s, role = %s, email_verified = %s,
                    tg_user_id = %s, tg_username = %s, display_name = %s
                WHERE id = %s
                """,
                (
                    user.email,
                    user.password_hash,
                    user.role,
                    user.email_verified,
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
            tg_user_id=tg_user_id, tg_username=tg_username, email_verified=False
        )

    def link_tg_to_user(
        self, user_id: int, tg_user_id: int, *, tg_username: Optional[str] = None
    ) -> None:
        with self._pool.connection() as conn:
            with conn.transaction():
                shadow = conn.execute(
                    "SELECT id FROM users WHERE tg_user_id = %s", (int(tg_user_id),)
                ).fetchone()
                if shadow is not None and int(shadow[0]) != user_id:
                    shadow_id = int(shadow[0])
                    conn.execute(
                        "UPDATE reports SET owner_id = %s WHERE owner_id = %s",
                        (user_id, shadow_id),
                    )
                    conn.execute("DELETE FROM users WHERE id = %s", (shadow_id,))
                conn.execute(
                    "UPDATE users SET tg_user_id = %s, tg_username = %s WHERE id = %s",
                    (int(tg_user_id), tg_username, user_id),
                )

    def count_users(self) -> int:
        with self._pool.connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------- email tokens
    def create_email_token(
        self, user_id: int, *, purpose: str, ttl_seconds: int
    ) -> EmailToken:
        token = secrets.token_urlsafe(32)
        now = time.time()
        exp = now + ttl_seconds
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO email_tokens (token, user_id, purpose, created_at, expires_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (token, user_id, purpose, now, exp),
            )
        return EmailToken(
            token=token, user_id=user_id, purpose=purpose, created_at=now, expires_at=exp
        )

    def consume_email_token(self, token: str, *, purpose: str) -> Optional[int]:
        with self._pool.connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    "SELECT user_id, purpose, expires_at FROM email_tokens "
                    "WHERE token = %s FOR UPDATE",
                    (token,),
                ).fetchone()
                if row is None:
                    return None
                conn.execute("DELETE FROM email_tokens WHERE token = %s", (token,))
                if str(row[1]) != purpose:
                    return None
                if float(row[2]) < time.time():
                    return None
                return int(row[0])

    # ----------------------------------------------------------------- settings
    def get_settings(self, owner_id: int) -> UserSettings:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT sources, search_limit, top_papers "
                "FROM user_settings WHERE owner_id = %s",
                (owner_id,),
            ).fetchone()
        if row is None:
            return UserSettings()
        try:
            sources = json.loads(row[0])
            if not isinstance(sources, list):
                sources = list(ALL_SOURCES)
        except (json.JSONDecodeError, TypeError):
            sources = list(ALL_SOURCES)
        sources = [s for s in sources if s in ALL_SOURCES]
        return UserSettings(
            sources=sources,
            search_limit=int(row[1]),
            top_papers=int(row[2]),
        )

    def save_settings(self, owner_id: int, s: UserSettings) -> None:
        payload = json.dumps(list(s.sources))
        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO user_settings
                    (owner_id, sources, search_limit, top_papers, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (owner_id) DO UPDATE SET
                    sources      = EXCLUDED.sources,
                    search_limit = EXCLUDED.search_limit,
                    top_papers   = EXCLUDED.top_papers,
                    updated_at   = EXCLUDED.updated_at
                """,
                (owner_id, payload, int(s.search_limit), int(s.top_papers), time.time()),
            )

    # ----------------------------------------------------------------- reports
    def add_report(
        self, owner_id: int, query: str, html: bytes, *, origin: str = "web"
    ) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO reports (owner_id, query, html, size_bytes, origin, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (owner_id, query, html, len(html), origin, time.time()),
            ).fetchone()
        return int(row[0])

    def count_reports(self, owner_id: int) -> int:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM reports WHERE owner_id = %s", (owner_id,)
            ).fetchone()
        return int(row[0]) if row else 0

    def list_reports(
        self, owner_id: int, *, offset: int, limit: int
    ) -> List[ReportRow]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, query, created_at, size_bytes, origin
                FROM reports
                WHERE owner_id = %s
                ORDER BY created_at DESC, id DESC
                LIMIT %s OFFSET %s
                """,
                (owner_id, int(limit), int(offset)),
            ).fetchall()
        return [
            ReportRow(
                id=int(r[0]),
                query=str(r[1]),
                created_at=float(r[2]),
                size_bytes=int(r[3]),
                origin=str(r[4]),
            )
            for r in rows
        ]

    def get_report_html(self, owner_id: int, report_id: int) -> Optional[bytes]:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT html FROM reports WHERE id = %s AND owner_id = %s",
                (int(report_id), owner_id),
            ).fetchone()
        if row is None:
            return None
        return bytes(row[0])

    # ------------------------------------------------------------------- stats
    def stats_global(self) -> dict:
        with self._pool.connection() as conn:
            users = int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
            reports = int(conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0])
            total_bytes = int(
                conn.execute(
                    "SELECT COALESCE(SUM(size_bytes), 0) FROM reports"
                ).fetchone()[0]
            )
            verified = int(
                conn.execute(
                    "SELECT COUNT(*) FROM users WHERE email_verified = TRUE"
                ).fetchone()[0]
            )
            tg_linked = int(
                conn.execute(
                    "SELECT COUNT(*) FROM users WHERE tg_user_id IS NOT NULL"
                ).fetchone()[0]
            )
            top_rows = conn.execute(
                "SELECT query, COUNT(*) AS n FROM reports "
                "GROUP BY query ORDER BY n DESC LIMIT 10"
            ).fetchall()
            by_origin = conn.execute(
                "SELECT origin, COUNT(*) FROM reports GROUP BY origin"
            ).fetchall()
        return {
            "users": users,
            "verified_users": verified,
            "tg_linked_users": tg_linked,
            "reports": reports,
            "total_bytes": total_bytes,
            "top_queries": [(str(r[0]), int(r[1])) for r in top_rows],
            "by_origin": {str(r[0]): int(r[1]) for r in by_origin},
        }

    def stats_for_owner(self, owner_id: int) -> dict:
        with self._pool.connection() as conn:
            reports = int(
                conn.execute(
                    "SELECT COUNT(*) FROM reports WHERE owner_id = %s", (owner_id,)
                ).fetchone()[0]
            )
            total_bytes = int(
                conn.execute(
                    "SELECT COALESCE(SUM(size_bytes), 0) FROM reports WHERE owner_id = %s",
                    (owner_id,),
                ).fetchone()[0]
            )
            top_rows = conn.execute(
                "SELECT query, COUNT(*) AS n FROM reports WHERE owner_id = %s "
                "GROUP BY query ORDER BY n DESC LIMIT 10",
                (owner_id,),
            ).fetchall()
        return {
            "reports": reports,
            "total_bytes": total_bytes,
            "top_queries": [(str(r[0]), int(r[1])) for r in top_rows],
        }
