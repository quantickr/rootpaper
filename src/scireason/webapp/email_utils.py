# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Отправка писем через SMTP с dev-fallback в лог.

Если ``settings.smtp_host`` не задан, письмо не отправляется, а ссылка
подтверждения пишется в лог — удобно для локальной разработки без почтового
сервера. Поддерживаются режимы ``ssl`` (порт 465), ``starttls`` (587) и
``none`` (25/2525) — совместимо, например, с Timeweb Cloud
(smtp.timeweb.ru: 465 SSL / 587 STARTTLS / 25|2525 без шифрования).
"""

import logging
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr
from typing import Optional

from ..config import settings

logger = logging.getLogger(__name__)


def smtp_enabled() -> bool:
    return bool((settings.smtp_host or "").strip())


def _send_via_smtp(to_email: str, subject: str, text_body: str, html_body: str) -> None:
    host = (settings.smtp_host or "").strip()
    port = int(settings.smtp_port)
    security = (settings.smtp_security or "ssl").strip().lower()
    user = settings.smtp_user
    password = settings.smtp_password
    from_addr = (settings.smtp_from or user or "").strip()

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr((settings.smtp_from_name, from_addr))
    msg["To"] = to_email
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    if security == "ssl":
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as server:
            if user and password:
                server.login(user, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=30) as server:
            if security == "starttls":
                server.starttls(context=ssl.create_default_context())
            if user and password:
                server.login(user, password)
            server.send_message(msg)


def send_verification_email(to_email: str, verify_url: str) -> bool:
    """Отправить письмо с подтверждением email. Возвращает True при отправке.

    При отсутствии SMTP-конфигурации ссылка пишется в лог (dev-режим) и
    возвращается False.
    """

    subject = "Подтверждение регистрации — top-papers-graph"
    text_body = (
        "Здравствуйте!\n\n"
        "Чтобы подтвердить адрес электронной почты, перейдите по ссылке:\n"
        f"{verify_url}\n\n"
        "Если вы не регистрировались, просто проигнорируйте это письмо."
    )
    html_body = (
        "<p>Здравствуйте!</p>"
        "<p>Чтобы подтвердить адрес электронной почты, нажмите кнопку:</p>"
        f'<p><a href="{verify_url}" '
        'style="display:inline-block;padding:10px 18px;background:#2563eb;'
        'color:#fff;border-radius:6px;text-decoration:none">Подтвердить email</a></p>'
        f'<p>Или откройте ссылку: <a href="{verify_url}">{verify_url}</a></p>'
        "<p style=\"color:#666;font-size:13px\">Если вы не регистрировались, "
        "просто проигнорируйте это письмо.</p>"
    )

    if not smtp_enabled():
        logger.warning(
            "SMTP не настроен — письмо не отправлено. Ссылка подтверждения для %s: %s",
            to_email,
            verify_url,
        )
        return False

    try:
        _send_via_smtp(to_email, subject, text_body, html_body)
        logger.info("Отправлено письмо подтверждения на %s", to_email)
        return True
    except Exception:  # pragma: no cover - зависит от внешнего SMTP
        logger.exception("Не удалось отправить письмо подтверждения на %s", to_email)
        # Fallback: логируем ссылку, чтобы регистрация не была заблокирована.
        logger.warning("Ссылка подтверждения для %s: %s", to_email, verify_url)
        return False


def _code_html(intro: str, code: str, note: str) -> str:
    return (
        f"<p>{intro}</p>"
        '<p style="font-size:32px;font-weight:700;letter-spacing:6px;'
        'font-family:monospace;color:#111;margin:18px 0">'
        f"{code}</p>"
        f'<p style="color:#666;font-size:13px">{note}</p>'
    )


def _send_code(
    to_email: str,
    *,
    code: str,
    subject: str,
    intro: str,
    note: str,
    log_label: str,
) -> bool:
    """Общий помощник: отправить письмо с 6-значным кодом.

    Возвращает True при успешной отправке. Если SMTP не настроен или отправка
    не удалась — код пишется в лог (dev-fallback) и возвращается False.
    """

    text_body = f"{intro}\n\nКод: {code}\n\n{note}"
    html_body = _code_html(intro, code, note)

    if not smtp_enabled():
        logger.warning(
            "SMTP не настроен — письмо не отправлено. %s для %s: %s",
            log_label,
            to_email,
            code,
        )
        return False

    try:
        _send_via_smtp(to_email, subject, text_body, html_body)
        logger.info("Отправлено письмо (%s) на %s", log_label, to_email)
        return True
    except Exception:  # pragma: no cover - зависит от внешнего SMTP
        logger.exception("Не удалось отправить письмо (%s) на %s", log_label, to_email)
        logger.warning("%s для %s: %s", log_label, to_email, code)
        return False


def send_verification_code(to_email: str, code: str) -> bool:
    """Отправить письмо с 6-значным кодом подтверждения email."""

    return _send_code(
        to_email,
        code=code,
        subject="Код подтверждения — RootPaper",
        intro="Здравствуйте! Ваш код подтверждения регистрации на RootPaper:",
        note=(
            "Введите этот код на странице подтверждения. Код действует "
            "ограниченное время. Если вы не регистрировались, "
            "просто проигнорируйте это письмо."
        ),
        log_label="Код подтверждения",
    )


def send_password_reset_code(to_email: str, code: str) -> bool:
    """Отправить письмо с 6-значным кодом для сброса пароля."""

    return _send_code(
        to_email,
        code=code,
        subject="Сброс пароля — RootPaper",
        intro="Здравствуйте! Ваш код для сброса пароля на RootPaper:",
        note=(
            "Введите этот код на странице сброса пароля вместе с новым "
            "паролем. Код действует ограниченное время. Если вы не запрашивали "
            "сброс пароля, просто проигнорируйте это письмо — ваш пароль не изменится."
        ),
        log_label="Код сброса пароля",
    )
