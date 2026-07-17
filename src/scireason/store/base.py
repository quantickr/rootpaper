# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Интерфейс общего хранилища и модели данных.

Хранилище держит:

* **пользователей сайта** (``users``) — email + bcrypt-хеш пароля, роль,
  флаг подтверждения почты, опциональная привязка Telegram (``tg_user_id``);
* **токены подтверждения email** (``email_tokens``);
* **настройки** (``user_settings``) и **историю отчётов** (``reports``) —
  ключ на ``owner_id`` (внутренний id пользователя сайта). Telegram-бот
  создаёт «теневого» пользователя по ``tg_user_id`` и работает через тот же
  ``owner_id``, поэтому отчёты из бота и с сайта живут в одной истории.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

# Доступные источники поиска (значения PaperSource в papers/service.py).
ALL_SOURCES: tuple[str, ...] = (
    "openalex",
    "semantic_scholar",
    "crossref",
    "arxiv",
    "pubmed",
    "europe_pmc",
    "biorxiv",
)

DEFAULT_SEARCH_LIMIT = 30
DEFAULT_TOP_PAPERS = 10

# Роли пользователей сайта.
ROLE_USER = "user"
ROLE_ADMIN = "admin"


@dataclass
class UserSettings:
    """Настройки одного пользователя (источники и лимиты пайплайна)."""

    sources: List[str] = field(default_factory=lambda: list(ALL_SOURCES))
    search_limit: int = DEFAULT_SEARCH_LIMIT
    top_papers: int = DEFAULT_TOP_PAPERS

    def sources_csv(self) -> str:
        """Строка для ``run_pipeline(sources=...)`` (``"all"`` если выбраны все)."""

        if not self.sources:
            return "all"
        if set(self.sources) == set(ALL_SOURCES):
            return "all"
        return ",".join(self.sources)


@dataclass
class ReportRow:
    """Одна запись истории отчётов (без самого HTML — только метаданные)."""

    id: int
    query: str
    created_at: float
    size_bytes: int
    origin: str = "web"  # "web" | "bot"


@dataclass
class RunJobRow:
    """Строка общего прогресса запуска (сайт и бот пишут в ``run_jobs``).

    Позволяет сайту показывать прогресс тем, запущенных из Telegram, и
    обновлять список отчётов без ручной перезагрузки страницы.
    """

    id: str
    owner_id: int
    query: str
    origin: str  # "web" | "bot"
    status: str  # "running" | "done" | "error"
    stage: str
    progress: float  # 0.0–1.0
    report_id: Optional[int]
    created_at: float
    updated_at: float


@dataclass
class User:
    """Пользователь сайта."""

    id: int
    email: Optional[str]
    password_hash: Optional[str]
    role: str = ROLE_USER
    email_verified: bool = False
    tg_user_id: Optional[int] = None
    tg_username: Optional[str] = None
    display_name: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN


@dataclass
class EmailToken:
    """Токен подтверждения email или сброса пароля."""

    token: str
    user_id: int
    purpose: str  # "verify" | "reset"
    created_at: float
    expires_at: float


class Store(ABC):
    """Абстрактный интерфейс хранилища.

    Реализации: :class:`SqliteStore`, :class:`PostgresStore`. Оба бэкенда
    делят одну и ту же логическую схему, что позволяет боту и сайту работать
    с одним хранилищем.
    """

    # ------------------------------------------------------------- users
    @abstractmethod
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
        """Создать пользователя и вернуть его (с присвоенным id)."""

    @abstractmethod
    def get_user(self, user_id: int) -> Optional[User]:
        ...

    @abstractmethod
    def get_user_by_email(self, email: str) -> Optional[User]:
        ...

    @abstractmethod
    def get_user_by_tg(self, tg_user_id: int) -> Optional[User]:
        ...

    @abstractmethod
    def update_user(self, user: User) -> None:
        """Сохранить изменённые поля пользователя."""

    @abstractmethod
    def get_or_create_tg_user(
        self, tg_user_id: int, *, tg_username: Optional[str] = None
    ) -> User:
        """Найти пользователя по ``tg_user_id`` или создать «теневого».

        Используется ботом: у Telegram-пользователя может ещё не быть
        веб-аккаунта, но настройки и отчёты нужно где-то хранить.
        """

    @abstractmethod
    def link_tg_to_user(
        self, user_id: int, tg_user_id: int, *, tg_username: Optional[str] = None
    ) -> None:
        """Привязать Telegram к существующему веб-аккаунту.

        Если под ``tg_user_id`` уже есть «теневой» пользователь, его данные
        (настройки и отчёты) переносятся на ``user_id``, после чего теневой
        удаляется. Это и есть «объединение аккаунтов».
        """

    @abstractmethod
    def count_users(self) -> int:
        ...

    # ------------------------------------------------- email tokens
    @abstractmethod
    def create_email_token(
        self, user_id: int, *, purpose: str, ttl_seconds: int
    ) -> EmailToken:
        ...

    @abstractmethod
    def consume_email_token(self, token: str, *, purpose: str) -> Optional[int]:
        """Проверить токен, удалить его и вернуть ``user_id`` при успехе."""

    @abstractmethod
    def create_email_code(
        self, user_id: int, *, purpose: str, ttl_seconds: int
    ) -> EmailToken:
        """Создать 6-значный числовой код для ``user_id`` и ``purpose``.

        Старые коды этого пользователя с тем же ``purpose`` удаляются, поэтому
        активным всегда остаётся один код. В отличие от
        :meth:`create_email_token`, значение — это короткий 6-значный код,
        поэтому проверять его нужно вместе с ``user_id`` (см.
        :meth:`consume_email_code`).
        """

    @abstractmethod
    def consume_email_code(
        self, user_id: int, code: str, *, purpose: str
    ) -> bool:
        """Проверить код для ``user_id``+``purpose``, удалить его и вернуть True.

        Возвращает False, если код неверный, с другим ``purpose`` или истёк.
        """

    # ------------------------------------------------------------- settings
    @abstractmethod
    def get_settings(self, owner_id: int) -> UserSettings:
        ...

    @abstractmethod
    def save_settings(self, owner_id: int, settings: UserSettings) -> None:
        ...

    def toggle_source(self, owner_id: int, source: str) -> UserSettings:
        if source not in ALL_SOURCES:
            return self.get_settings(owner_id)
        s = self.get_settings(owner_id)
        if source in s.sources:
            s.sources = [x for x in s.sources if x != source]
        else:
            chosen = set(s.sources) | {source}
            s.sources = [x for x in ALL_SOURCES if x in chosen]
        self.save_settings(owner_id, s)
        return s

    def set_search_limit(self, owner_id: int, value: int) -> UserSettings:
        s = self.get_settings(owner_id)
        s.search_limit = max(1, int(value))
        self.save_settings(owner_id, s)
        return s

    def set_top_papers(self, owner_id: int, value: int) -> UserSettings:
        s = self.get_settings(owner_id)
        s.top_papers = max(1, int(value))
        self.save_settings(owner_id, s)
        return s

    # ------------------------------------------------------------- reports
    @abstractmethod
    def add_report(
        self, owner_id: int, query: str, html: bytes, *, origin: str = "web"
    ) -> int:
        ...

    @abstractmethod
    def count_reports(self, owner_id: int) -> int:
        ...

    @abstractmethod
    def list_reports(
        self, owner_id: int, *, offset: int, limit: int
    ) -> List[ReportRow]:
        ...

    @abstractmethod
    def get_report_html(self, owner_id: int, report_id: int) -> Optional[bytes]:
        """HTML отчёта, только если он принадлежит ``owner_id``."""

    # ----------------------------------------------------------- report shares
    @abstractmethod
    def create_report_share(self, report_id: int) -> str:
        """Вернуть публичный share-токен отчёта, создав его при необходимости.

        Идемпотентно: если у отчёта уже есть токен — возвращается он же.
        """

    @abstractmethod
    def get_report_html_by_share(self, token: str) -> Optional[bytes]:
        """HTML отчёта по публичному share-токену (без проверки владельца)."""

    # --------------------------------------------------------------- run jobs
    @abstractmethod
    def create_run_job(
        self, job_id: str, owner_id: int, query: str, *, origin: str = "web"
    ) -> None:
        """Зарегистрировать новый запуск (статус running, progress 0)."""

    @abstractmethod
    def update_run_job(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        progress: Optional[float] = None,
        report_id: Optional[int] = None,
    ) -> None:
        """Обновить прогресс/статус запуска (частичное обновление)."""

    @abstractmethod
    def get_run_job(self, job_id: str) -> Optional[RunJobRow]:
        """Одна строка прогресса по её id (или None)."""

    @abstractmethod
    def list_active_run_jobs(self, owner_id: int) -> List[RunJobRow]:
        """Незавершённые запуски владельца (status='running')."""

    # ------------------------------------------------------------- stats
    @abstractmethod
    def stats_global(self) -> dict:
        """Общая статистика (для админ-дашборда)."""

    @abstractmethod
    def stats_for_owner(self, owner_id: int) -> dict:
        """Статистика по одному пользователю."""
