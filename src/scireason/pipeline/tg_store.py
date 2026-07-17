# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Совместимая обёртка хранилища для Telegram-бота.

Раньше этот модуль сам работал с SQLite. Теперь хранилище общее с веб-сайтом
(:mod:`scireason.store`): бот и сайт пишут в одну БД (SQLite по умолчанию или
PostgreSQL, если задан ``DATABASE_URL``).

``TgStore`` остаётся тонкой обёрткой с прежним API (ключ — Telegram
``user_id``). Внутри он транслирует ``user_id`` в внутренний ``owner_id``
пользователя сайта через :meth:`Store.get_or_create_tg_user`, поэтому отчёты
и настройки из бота и с сайта живут в одной истории и легко объединяются при
привязке аккаунта.

Публичные имена (``UserSettings``, ``ReportRow``, ``ALL_SOURCES`` и т.д.)
реэкспортируются из :mod:`scireason.store.base` для обратной совместимости.
"""

from pathlib import Path
from typing import List, Optional

from ..store import get_store
from ..store.base import (  # re-export для обратной совместимости
    ALL_SOURCES,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_TOP_PAPERS,
    ReportRow,
    Store,
    UserSettings,
)
from ..store.sqlite_store import DEFAULT_DB_PATH

__all__ = [
    "ALL_SOURCES",
    "DEFAULT_SEARCH_LIMIT",
    "DEFAULT_TOP_PAPERS",
    "DEFAULT_DB_PATH",
    "UserSettings",
    "ReportRow",
    "TgStore",
]


class TgStore:
    """Обёртка над общим :class:`Store` с Telegram-ориентированным API.

    ``db_path`` сохранён для обратной совместимости, но по умолчанию
    используется общий синглтон хранилища (см. :func:`scireason.store.get_store`),
    что гарантирует единое хранилище с веб-сайтом.
    """

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        *,
        store: Optional[Store] = None,
    ) -> None:
        self._store: Store = store if store is not None else get_store()

    # ------------------------------------------------------- owner resolution
    def _owner_id(self, tg_user_id: int, tg_username: Optional[str] = None) -> int:
        user = self._store.get_or_create_tg_user(
            int(tg_user_id), tg_username=tg_username
        )
        return user.id

    def ensure_tg_user(
        self, tg_user_id: int, tg_username: Optional[str] = None
    ) -> None:
        """Создать/обновить теневого Telegram-пользователя (с username).

        Вызывается ботом при первом контакте, чтобы у пользователя был
        ``tg_username`` — это упрощает объединение аккаунтов на сайте.
        """

        self._store.get_or_create_tg_user(int(tg_user_id), tg_username=tg_username)

    # ------------------------------------------------------------------ settings
    def get_settings(self, user_id: int) -> UserSettings:
        return self._store.get_settings(self._owner_id(user_id))

    def save_settings(self, user_id: int, s: UserSettings) -> None:
        self._store.save_settings(self._owner_id(user_id), s)

    def toggle_source(self, user_id: int, source: str) -> UserSettings:
        return self._store.toggle_source(self._owner_id(user_id), source)

    def set_search_limit(self, user_id: int, value: int) -> UserSettings:
        return self._store.set_search_limit(self._owner_id(user_id), value)

    def set_top_papers(self, user_id: int, value: int) -> UserSettings:
        return self._store.set_top_papers(self._owner_id(user_id), value)

    # ------------------------------------------------------------------- reports
    def add_report(self, user_id: int, query: str, html: bytes) -> int:
        return self._store.add_report(
            self._owner_id(user_id), query, html, origin="bot"
        )

    def count_reports(self, user_id: int) -> int:
        return self._store.count_reports(self._owner_id(user_id))

    def list_reports(
        self, user_id: int, *, offset: int, limit: int
    ) -> List[ReportRow]:
        return self._store.list_reports(
            self._owner_id(user_id), offset=offset, limit=limit
        )

    def get_report_html(self, user_id: int, report_id: int) -> Optional[bytes]:
        return self._store.get_report_html(self._owner_id(user_id), report_id)

    def create_report_share(self, report_id: int) -> str:
        """Публичный share-токен отчёта для ссылки на сайт (из бота)."""

        return self._store.create_report_share(report_id)
