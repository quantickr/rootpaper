# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Роуты просмотра истории отчётов и настроек пользователя."""

from datetime import datetime

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from ...store.base import ALL_SOURCES
from ..deps import current_user, get_db, render

router = APIRouter()

_PAGE_SIZE = 10

_SOURCE_TITLES = {
    "openalex": "OpenAlex",
    "semantic_scholar": "Semantic Scholar",
    "crossref": "Crossref",
    "arxiv": "arXiv",
    "pubmed": "PubMed",
    "europe_pmc": "Europe PMC",
    "biorxiv": "bioRxiv",
}


@router.get("/reports")
def reports_list(request: Request, page: int = Query(1, ge=1)):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    db = get_db()
    total = db.count_reports(user.id)
    offset = (page - 1) * _PAGE_SIZE
    rows = db.list_reports(user.id, offset=offset, limit=_PAGE_SIZE)
    items = [
        {
            "id": r.id,
            "query": r.query,
            "when": datetime.fromtimestamp(r.created_at).strftime("%Y-%m-%d %H:%M"),
            "size_kb": round(r.size_bytes / 1024, 1),
            "origin": r.origin,
        }
        for r in rows
    ]
    pages = max(1, (total + _PAGE_SIZE - 1) // _PAGE_SIZE)
    return render(
        request,
        "reports.html",
        items=items,
        total=total,
        page=page,
        pages=pages,
    )


@router.get("/reports/{report_id}/view", response_class=HTMLResponse)
def report_view(request: Request, report_id: int):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    html = get_db().get_report_html(user.id, report_id)
    if html is None:
        resp = render(
            request,
            "message.html",
            title="Отчёт не найден",
            message="Такого отчёта нет или он принадлежит другому пользователю.",
        )
        resp.status_code = 404
        return resp
    # Отдаём сам самодостаточный HTML-отчёт как есть.
    return HTMLResponse(content=html.decode("utf-8", errors="replace"))


@router.get("/reports/{report_id}/download")
def report_download(request: Request, report_id: int):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    html = get_db().get_report_html(user.id, report_id)
    if html is None:
        resp = render(
            request,
            "message.html",
            title="Отчёт не найден",
            message="Такого отчёта нет или он принадлежит другому пользователю.",
        )
        resp.status_code = 404
        return resp
    return Response(
        content=html,
        media_type="text/html",
        headers={
            "Content-Disposition": f'attachment; filename="report_{report_id}.html"'
        },
    )


@router.get("/r/{token}", response_class=HTMLResponse)
def report_shared_view(request: Request, token: str):
    """Публичный просмотр отчёта по секретному share-токену (без логина).

    Доступ ограничен «секретом в URL»: угадать unguessable token нельзя.
    Используется ссылками из Telegram-бота.
    """

    html = get_db().get_report_html_by_share(token)
    if html is None:
        resp = render(
            request,
            "message.html",
            title="Отчёт не найден",
            message="Ссылка недействительна или отчёт был удалён.",
        )
        resp.status_code = 404
        return resp
    return HTMLResponse(content=html.decode("utf-8", errors="replace"))


@router.get("/settings")
def settings_form(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    s = get_db().get_settings(user.id)
    sources = [
        {"key": k, "title": _SOURCE_TITLES.get(k, k), "on": k in s.sources}
        for k in ALL_SOURCES
    ]
    return render(
        request,
        "settings.html",
        sources=sources,
        search_limit=s.search_limit,
        top_papers=s.top_papers,
    )


@router.post("/settings")
async def settings_submit(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    db = get_db()
    form = await request.form()
    # Чекбоксы источников приходят как несколько полей source=<key>.
    chosen = [v for v in form.getlist("source") if v in ALL_SOURCES]
    ordered = [k for k in ALL_SOURCES if k in set(chosen)]
    s = db.get_settings(user.id)
    s.sources = ordered
    try:
        s.search_limit = max(1, int(form.get("search_limit", s.search_limit)))
    except (TypeError, ValueError):
        pass
    try:
        s.top_papers = max(1, int(form.get("top_papers", s.top_papers)))
    except (TypeError, ValueError):
        pass
    db.save_settings(user.id, s)
    return RedirectResponse("/settings", status_code=303)
