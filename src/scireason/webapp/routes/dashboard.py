# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Дашборд со статистикой: админ видит общую, обычный юзер — свою."""

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from ..deps import current_user, get_db, render

router = APIRouter()


def _fmt_mb(nbytes: int) -> float:
    return round(nbytes / (1024 * 1024), 2)


@router.get("/dashboard")
def dashboard(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    db = get_db()
    if user.is_admin:
        stats = db.stats_global()
        return render(
            request,
            "dashboard_admin.html",
            stats=stats,
            total_mb=_fmt_mb(stats["total_bytes"]),
        )
    stats = db.stats_for_owner(user.id)
    return render(
        request,
        "dashboard_user.html",
        stats=stats,
        total_mb=_fmt_mb(stats["total_bytes"]),
    )
