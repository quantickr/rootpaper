# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Общий слой хранилища для Telegram-бота и веб-сайта.

Бот и сайт — два разных процесса, которые пишут в одну БД (настройки,
пользователи, история отчётов). Модуль предоставляет единый интерфейс
:class:`~scireason.store.base.Store` и две реализации:

* :class:`~scireason.store.sqlite_store.SqliteStore` — по умолчанию (офлайн,
  один файл, ничего не нужно поднимать);
* :class:`~scireason.store.postgres_store.PostgresStore` — когда задан
  ``DATABASE_URL`` (надёжнее при конкурентной записи из бота и сайта).

Выбор бэкенда — через :func:`~scireason.store.factory.get_store`.
"""

from .base import (
    EmailToken,
    ReportRow,
    Store,
    User,
    UserSettings,
    ALL_SOURCES,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_TOP_PAPERS,
)
from .factory import get_store

__all__ = [
    "Store",
    "User",
    "UserSettings",
    "ReportRow",
    "EmailToken",
    "ALL_SOURCES",
    "DEFAULT_SEARCH_LIMIT",
    "DEFAULT_TOP_PAPERS",
    "get_store",
]
