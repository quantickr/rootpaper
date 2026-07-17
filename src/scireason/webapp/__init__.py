# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Веб-сайт (FastAPI + Jinja2) поверх общего хранилища.

Возможности:

* регистрация по email/паролю с подтверждением через SMTP;
* вход через Telegram Login Widget и объединение аккаунтов;
* просмотр истории отчётов (общая с ботом);
* запуск пайплайна из браузера;
* дашборд со статистикой (роль ``admin`` видит общую).
"""

from .app import create_app

__all__ = ["create_app"]
