# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Общие зависимости FastAPI: сессия, текущий пользователь, шаблоны."""

from pathlib import Path
from typing import Optional

from fastapi import Request
from fastapi.templating import Jinja2Templates

from ..config import settings
from ..store import Store, get_store
from ..store.base import User
from . import security

SESSION_COOKIE = "tpg_session"

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def get_db() -> Store:
    return get_store()


def current_user(request: Request) -> Optional[User]:
    """Пользователь из cookie-сессии или ``None``."""

    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    uid = security.read_session_token(settings.web_secret_key, token)
    if uid is None:
        return None
    return get_store().get_user(uid)


def set_session_cookie(response, user_id: int) -> None:
    token = security.make_session_token(settings.web_secret_key, user_id)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=security.SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=bool(settings.web_session_https_only),
    )


def clear_session_cookie(response) -> None:
    response.delete_cookie(SESSION_COOKIE)


def render(request: Request, template: str, **ctx):
    """Отрисовать шаблон, всегда добавляя ``user`` и ``settings`` в контекст."""

    base = {
        "user": current_user(request),
        "cfg": settings,
    }
    base.update(ctx)
    return templates.TemplateResponse(request, template, base)
