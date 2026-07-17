# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Выбор бэкенда хранилища и общий синглтон.

Логика:

* если задан ``DATABASE_URL`` (или ``settings.database_url``) и он указывает на
  PostgreSQL — используется :class:`PostgresStore`;
* иначе — :class:`SqliteStore` (офлайн-режим по умолчанию, один файл).

Синглтон гарантирует, что бот и веб-сайт в одном процессе (а также разные
запросы к сайту) делят одно хранилище/пул соединений.
"""

import threading
from typing import Optional

from .base import Store

_lock = threading.Lock()
_instance: Optional[Store] = None


def _build_store() -> Store:
    from ..config import settings

    dsn = (settings.database_url or "").strip()
    if dsn and (dsn.startswith("postgres://") or dsn.startswith("postgresql://")):
        from .postgres_store import PostgresStore

        return PostgresStore(dsn)

    from .sqlite_store import DEFAULT_DB_PATH, SqliteStore

    db_path = (settings.sqlite_db_path or str(DEFAULT_DB_PATH)).strip()
    return SqliteStore(db_path)


def get_store() -> Store:
    """Вернуть общий экземпляр хранилища (создаётся лениво один раз)."""

    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = _build_store()
    return _instance


def reset_store_for_tests(store: Optional[Store] = None) -> None:
    """Сбросить/подменить синглтон (только для тестов)."""

    global _instance
    with _lock:
        _instance = store
