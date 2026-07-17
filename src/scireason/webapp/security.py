# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Хеширование паролей, подпись сессий и проверка Telegram Login Widget.

* Пароли хешируются bcrypt (напрямую через пакет ``bcrypt``). Прямой вызов
  выбран вместо ``passlib`` из-за несовместимости ``passlib<=1.7.4`` с
  ``bcrypt>=4.1`` (passlib падает при инициализации бэкенда).
* Cookie-сессия — подписанный ``itsdangerous`` JSON (без серверного стораджа
  сессий: держим только ``uid`` и время).
* Данные Telegram Login Widget проверяются по HMAC-SHA256 с ключом
  ``sha256(bot_token)`` (алгоритм из документации Telegram).
"""

import hashlib
import hmac
import re
import time
from typing import Optional

import bcrypt
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Куки-сессия живёт 30 дней.
SESSION_MAX_AGE = 30 * 24 * 3600
_SESSION_SALT = "tpg-web-session"

# bcrypt использует только первые 72 байта пароля.
_BCRYPT_MAX_BYTES = 72


def _pw_bytes(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


# ------------------------------------------------------------------ passwords
def hash_password(password: str) -> str:
    return bcrypt.hashpw(_pw_bytes(password), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: Optional[str]) -> bool:
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(_pw_bytes(password), password_hash.encode("ascii"))
    except Exception:
        return False


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(email.strip()))


def password_problem(password: str) -> Optional[str]:
    """Вернуть текст ошибки, если пароль слишком слабый, иначе ``None``."""

    if len(password) < 8:
        return "Пароль должен быть не короче 8 символов."
    if password.isdigit() or password.isalpha():
        return "Пароль должен содержать и буквы, и цифры."
    return None


# ------------------------------------------------------------------- sessions
def _serializer(secret_key: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret_key, salt=_SESSION_SALT)


def make_session_token(secret_key: str, user_id: int) -> str:
    return _serializer(secret_key).dumps({"uid": int(user_id)})


def read_session_token(secret_key: str, token: str) -> Optional[int]:
    try:
        data = _serializer(secret_key).loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired, Exception):
        return None
    uid = data.get("uid")
    return int(uid) if uid is not None else None


# ------------------------------------------------------- Telegram Login Widget
def verify_telegram_login(data: dict, bot_token: str, *, max_age: int = 86400) -> bool:
    """Проверить подпись данных Telegram Login Widget.

    ``data`` — словарь query-параметров от виджета (id, first_name, username,
    photo_url, auth_date, hash). Возвращает True, если подпись валидна и
    ``auth_date`` не старше ``max_age`` секунд.
    """

    if not bot_token:
        return False
    received_hash = data.get("hash")
    if not received_hash:
        return False

    # Строка проверки: все поля кроме hash, отсортированы, "key=value\n".
    pairs = [f"{k}={v}" for k, v in sorted(data.items()) if k != "hash"]
    check_string = "\n".join(pairs)

    secret_key = hashlib.sha256(bot_token.encode("utf-8")).digest()
    computed = hmac.new(
        secret_key, check_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(computed, str(received_hash)):
        return False

    try:
        auth_date = int(data.get("auth_date", "0"))
    except (TypeError, ValueError):
        return False
    if auth_date <= 0 or (time.time() - auth_date) > max_age:
        return False
    return True
