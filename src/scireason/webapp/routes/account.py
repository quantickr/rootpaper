# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Профиль пользователя: привязка Telegram, объединение аккаунтов."""

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from ...config import settings
from ..deps import current_user, get_db, render

router = APIRouter()


@router.get("/account")
def account(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    return render(
        request,
        "account.html",
        bot_username=(settings.telegram_bot_username or "").lstrip("@"),
    )


@router.post("/account/unlink-telegram")
def unlink_telegram(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    # Отвязать Telegram можно только у аккаунта с email (иначе потеряется вход).
    if user.email:
        user.tg_user_id = None
        user.tg_username = None
        get_db().update_user(user)
    return RedirectResponse("/account", status_code=303)
