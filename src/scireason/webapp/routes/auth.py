# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Роуты аутентификации: регистрация, вход, email-подтверждение, Telegram."""

import logging
from typing import Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from ...config import settings
from ...store.base import ROLE_ADMIN, ROLE_USER
from .. import security
from ..deps import (
    clear_session_cookie,
    current_user,
    get_db,
    render,
    set_session_cookie,
)
from ..email_utils import send_password_reset_code, send_verification_code

logger = logging.getLogger(__name__)
router = APIRouter()

# Минимальный интервал между запросами кода (verify/reset) на один аккаунт.
EMAIL_CODE_COOLDOWN_SECONDS = 60


def _cooldown_message(remaining: float) -> str:
    secs = max(1, int(round(remaining)))
    return (
        f"Код уже отправлен. Запросить новый можно через {secs} сек. "
        "Проверьте почту (в т.ч. спам)."
    )


def _admin_emails() -> set[str]:
    raw = settings.admin_emails or ""
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


# ---------------------------------------------------------------- registration
@router.get("/register")
def register_form(request: Request):
    if current_user(request):
        return RedirectResponse("/reports", status_code=303)
    return render(request, "register.html")


@router.post("/register")
def register_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    password2: str = Form(...),
):
    db = get_db()
    email = email.strip().lower()

    if not security.is_valid_email(email):
        return render(request, "register.html", error="Некорректный email.", email=email)
    if password != password2:
        return render(
            request, "register.html", error="Пароли не совпадают.", email=email
        )
    problem = security.password_problem(password)
    if problem:
        return render(request, "register.html", error=problem, email=email)
    if db.get_user_by_email(email) is not None:
        return render(
            request,
            "register.html",
            error="Пользователь с таким email уже существует.",
            email=email,
        )

    role = ROLE_ADMIN if email in _admin_emails() else ROLE_USER
    user = db.create_user(
        email=email,
        password_hash=security.hash_password(password),
        role=role,
        email_verified=False,
    )

    code = db.create_email_code(
        user.id, purpose="verify", ttl_seconds=settings.email_verify_ttl_seconds
    )
    sent = send_verification_code(email, code.token)

    return render(
        request,
        "verify_code.html",
        email=email,
        sent=sent,
        info=(
            f"Мы отправили 6-значный код подтверждения на {email}. "
            "Введите его ниже."
        ),
    )


# ------------------------------------------------------------- email code verify
@router.get("/verify")
def verify_form(request: Request, email: str = ""):
    return render(request, "verify_code.html", email=email.strip().lower())


@router.post("/verify")
def verify_submit(
    request: Request,
    email: str = Form(...),
    code: str = Form(...),
):
    db = get_db()
    email = email.strip().lower()
    code = (code or "").strip()
    user = db.get_user_by_email(email)
    if user is None:
        return render(
            request,
            "verify_code.html",
            email=email,
            error="Аккаунт не найден. Зарегистрируйтесь заново.",
        )
    if user.email_verified:
        resp = RedirectResponse("/reports", status_code=303)
        set_session_cookie(resp, user.id)
        return resp
    if not db.consume_email_code(user.id, code, purpose="verify"):
        return render(
            request,
            "verify_code.html",
            email=email,
            error="Неверный или устаревший код. Проверьте код или запросите новый.",
        )
    user.email_verified = True
    db.update_user(user)
    resp = RedirectResponse("/reports", status_code=303)
    set_session_cookie(resp, user.id)
    return resp


@router.post("/verify/resend")
def verify_resend(request: Request, email: str = Form(...)):
    db = get_db()
    email = email.strip().lower()
    user = db.get_user_by_email(email)
    if user is not None and not user.email_verified:
        remaining = db.email_code_cooldown_remaining(
            user.id, purpose="verify", cooldown_seconds=EMAIL_CODE_COOLDOWN_SECONDS
        )
        if remaining > 0:
            return render(
                request,
                "verify_code.html",
                email=email,
                error=_cooldown_message(remaining),
            )
        code = db.create_email_code(
            user.id, purpose="verify", ttl_seconds=settings.email_verify_ttl_seconds
        )
        send_verification_code(email, code.token)
    # Нейтральное сообщение независимо от наличия аккаунта.
    return render(
        request,
        "verify_code.html",
        email=email,
        info="Если аккаунт существует и не подтверждён, мы отправили новый код.",
    )


# ------------------------------------------------------------------------ login
@router.get("/login")
def login_form(request: Request):
    if current_user(request):
        return RedirectResponse("/reports", status_code=303)
    return render(request, "login.html")


@router.post("/login")
def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    db = get_db()
    email = email.strip().lower()
    user = db.get_user_by_email(email)
    if user is None or not security.verify_password(password, user.password_hash):
        return render(
            request, "login.html", error="Неверный email или пароль.", email=email
        )
    if not user.email_verified:
        # Отправим свежий код и сразу покажем форму ввода — с учётом кулдауна.
        remaining = db.email_code_cooldown_remaining(
            user.id, purpose="verify", cooldown_seconds=EMAIL_CODE_COOLDOWN_SECONDS
        )
        if remaining > 0:
            return render(
                request,
                "verify_code.html",
                email=email,
                error=_cooldown_message(remaining),
            )
        code = db.create_email_code(
            user.id, purpose="verify", ttl_seconds=settings.email_verify_ttl_seconds
        )
        send_verification_code(email, code.token)
        return render(
            request,
            "verify_code.html",
            email=email,
            info="Email не подтверждён. Мы отправили новый код на вашу почту.",
        )
    resp = RedirectResponse("/reports", status_code=303)
    set_session_cookie(resp, user.id)
    return resp


@router.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    clear_session_cookie(resp)
    return resp


# --------------------------------------------------------- password reset (forgot)
# Нейтральное сообщение, чтобы не раскрывать, существует ли аккаунт.
_RESET_NEUTRAL = (
    "Если аккаунт с таким email существует, мы отправили на него 6-значный "
    "код для сброса пароля."
)


@router.get("/forgot")
def forgot_form(request: Request):
    if current_user(request):
        return RedirectResponse("/reports", status_code=303)
    return render(request, "forgot.html")


@router.post("/forgot")
def forgot_submit(request: Request, email: str = Form(...)):
    db = get_db()
    email = email.strip().lower()
    if security.is_valid_email(email):
        user = db.get_user_by_email(email)
        if user is not None and user.email:
            # Кулдаун: если код уже слали недавно, молча пропускаем повторную
            # отправку (сообщение остаётся нейтральным, без раскрытия аккаунта).
            remaining = db.email_code_cooldown_remaining(
                user.id, purpose="reset", cooldown_seconds=EMAIL_CODE_COOLDOWN_SECONDS
            )
            if remaining <= 0:
                code = db.create_email_code(
                    user.id,
                    purpose="reset",
                    ttl_seconds=settings.email_verify_ttl_seconds,
                )
                send_password_reset_code(email, code.token)
    # Всегда ведём на форму ввода кода с нейтральным сообщением.
    return render(request, "reset.html", email=email, info=_RESET_NEUTRAL)


@router.get("/reset")
def reset_form(request: Request, email: str = ""):
    if current_user(request):
        return RedirectResponse("/reports", status_code=303)
    return render(request, "reset.html", email=email.strip().lower())


@router.post("/reset")
def reset_submit(
    request: Request,
    email: str = Form(...),
    code: str = Form(...),
    password: str = Form(...),
    password2: str = Form(...),
):
    db = get_db()
    email = email.strip().lower()
    code = (code or "").strip()

    if password != password2:
        return render(
            request, "reset.html", email=email, error="Пароли не совпадают."
        )
    problem = security.password_problem(password)
    if problem:
        return render(request, "reset.html", email=email, error=problem)

    user = db.get_user_by_email(email)
    if user is None or not db.consume_email_code(user.id, code, purpose="reset"):
        return render(
            request,
            "reset.html",
            email=email,
            error="Неверный или устаревший код. Запросите новый на странице «Забыли пароль?».",
        )

    user.password_hash = security.hash_password(password)
    # Успешный сброс подтверждает владение почтой.
    user.email_verified = True
    db.update_user(user)
    resp = RedirectResponse("/reports", status_code=303)
    set_session_cookie(resp, user.id)
    return resp


# --------------------------------------------------------------------- telegram
def _collect_tg_data(request: Request) -> dict:
    """Собрать поля Telegram Login Widget из query-параметров."""

    keys = (
        "id",
        "first_name",
        "last_name",
        "username",
        "photo_url",
        "auth_date",
        "hash",
    )
    data = {}
    for k in keys:
        v = request.query_params.get(k)
        if v is not None:
            data[k] = v
    return data


@router.get("/auth/telegram")
def telegram_callback(request: Request):
    """Callback Telegram Login Widget.

    * Если пользователь уже вошёл в веб-аккаунт — привязывает Telegram к нему
      (объединение аккаунтов).
    * Иначе — входит по Telegram (создаёт/находит пользователя).
    """

    db = get_db()
    bot_token = (settings.telegram_bot_token or "").strip()
    data = _collect_tg_data(request)

    if not security.verify_telegram_login(data, bot_token):
        return render(
            request,
            "message.html",
            title="Ошибка входа через Telegram",
            message="Не удалось проверить подпись Telegram. Попробуйте ещё раз.",
        )

    tg_user_id = int(data["id"])
    tg_username = data.get("username")

    logged_in = current_user(request)
    if logged_in is not None:
        # Привязка Telegram к текущему веб-аккаунту (объединение).
        other = db.get_user_by_tg(tg_user_id)
        if other is not None and other.id != logged_in.id and other.email:
            # Telegram уже привязан к другому полноценному аккаунту.
            return render(
                request,
                "message.html",
                title="Telegram уже привязан",
                message="Этот Telegram уже связан с другим аккаунтом. "
                "Сначала отвяжите его там.",
            )
        db.link_tg_to_user(logged_in.id, tg_user_id, tg_username=tg_username)
        return RedirectResponse("/account", status_code=303)

    # Вход/регистрация через Telegram.
    user = db.get_or_create_tg_user(tg_user_id, tg_username=tg_username)
    if not user.display_name and data.get("first_name"):
        user.display_name = data.get("first_name")
        db.update_user(user)
    resp = RedirectResponse("/reports", status_code=303)
    set_session_cookie(resp, user.id)
    return resp
