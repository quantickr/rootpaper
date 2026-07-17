# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Роуты запуска пайплайна из браузера и опроса статуса задачи."""

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse, RedirectResponse

from ..deps import current_user, render
from ..jobs import get_job_manager

router = APIRouter()


@router.get("/run")
def run_form(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    jobs = get_job_manager().list_for_owner(user.id)
    return render(request, "run.html", jobs=jobs[:10])


@router.post("/run")
def run_submit(request: Request, query: str = Form(...)):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    query = (query or "").strip()
    if not query:
        return render(
            request,
            "run.html",
            error="Введите тему запроса.",
            jobs=get_job_manager().list_for_owner(user.id)[:10],
        )
    job = get_job_manager().submit(user.id, query)
    return RedirectResponse(f"/run/{job.id}", status_code=303)


@router.get("/run/{job_id}")
def run_status_page(request: Request, job_id: str):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    job = get_job_manager().get(job_id)
    if job is None or job.owner_id != user.id:
        return render(
            request,
            "message.html",
            title="Задача не найдена",
            message="Такой задачи нет или она принадлежит другому пользователю.",
        )
    return render(request, "run_status.html", job=job)


@router.get("/run/{job_id}/state")
def run_status_json(request: Request, job_id: str):
    """JSON-эндпоинт для опроса статуса (используется страницей run_status)."""

    user = current_user(request)
    if user is None:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    job = get_job_manager().get(job_id)
    if job is None or job.owner_id != user.id:
        return JSONResponse({"error": "not_found"}, status_code=404)
    return JSONResponse(
        {
            "id": job.id,
            "status": job.status,
            "message": job.message,
            "report_id": job.report_id,
        }
    )
