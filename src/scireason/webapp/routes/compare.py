# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Роуты сравнения нескольких статей (форма + запуск задачи)."""

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from ..deps import current_user, render
from ..jobs import get_job_manager

router = APIRouter()


def _count_lines(text: str) -> int:
    return len([x for x in (text or "").splitlines() if x.strip()])


@router.get("/compare")
def compare_form(request: Request):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    jobs = [j for j in get_job_manager().list_for_owner(user.id) if j.kind == "compare"]
    return render(request, "compare.html", jobs=jobs[:10])


@router.post("/compare")
def compare_submit(request: Request, inputs: str = Form(...)):
    user = current_user(request)
    if user is None:
        return RedirectResponse("/login", status_code=303)
    inputs = (inputs or "").strip()
    if _count_lines(inputs) < 2:
        jobs = [
            j for j in get_job_manager().list_for_owner(user.id) if j.kind == "compare"
        ]
        return render(
            request,
            "compare.html",
            error="Нужно минимум 2 статьи — по одной ссылке/DOI/arXiv-ID на строку.",
            inputs=inputs,
            jobs=jobs[:10],
        )
    job = get_job_manager().submit_compare(user.id, inputs)
    return RedirectResponse(f"/run/{job.id}", status_code=303)
