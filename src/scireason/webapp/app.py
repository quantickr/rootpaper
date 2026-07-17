# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Сборка FastAPI-приложения веб-сайта.

Важно: без ``from __future__ import annotations``. Иначе строковые аннотации
inline-эндпоинтов (``request: Request``) не резолвятся FastAPI и ``Request``
ошибочно трактуется как query-параметр.
"""

from pathlib import Path

from ..config import settings


def create_app():
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import RedirectResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Web extra is not installed. Install with: pip install -e '.[web]'"
        ) from e

    from .deps import current_user, render
    from .routes import account, auth, dashboard, pipeline, reports

    app = FastAPI(
        title="rootpaper — сайт",
        version="0.1.0",
        description="Веб-интерфейс с регистрацией, историей отчётов и запуском пайплайна.",
    )

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    def index(request: Request):
        if current_user(request):
            return RedirectResponse("/reports", status_code=303)
        return render(request, "index.html")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    app.include_router(auth.router)
    app.include_router(reports.router)
    app.include_router(pipeline.router)
    app.include_router(dashboard.router)
    app.include_router(account.router)

    return app


app = create_app()
