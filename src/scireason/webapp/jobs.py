# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""In-memory менеджер фоновых задач пайплайна для веб-сайта.

Пайплайн (:func:`scireason.pipeline.e2e.run_pipeline`) синхронный и долгий,
поэтому запускаем его в пуле потоков. Статус задач держим в памяти процесса —
для простоты и офлайн-режима этого достаточно (готовый отчёт всё равно
сохраняется в БД и доступен в истории даже после перезапуска).
"""

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    owner_id: int
    query: str
    status: str = "queued"  # queued | running | done | error
    message: str = ""
    report_id: Optional[int] = None
    progress: float = 0.0  # 0.0–1.0, обновляется колбэком пайплайна
    stage: str = ""  # человекочитаемая метка текущего этапа
    created_at: float = field(default_factory=time.time)
    # kind="search" — обычный пайплайн; kind="compare" — сравнение статей.
    kind: str = "search"
    # Для сравнения: сырой ввод пользователя (ссылки/DOI/arXiv-ID по строкам).
    inputs: str = ""


class JobManager:
    """Пул потоков + реестр задач в памяти."""

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers))
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, owner_id: int, query: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], owner_id=owner_id, query=query)
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job.id)
        return job

    def submit_compare(self, owner_id: int, inputs: str) -> Job:
        """Поставить задачу сравнения статей (ссылки/DOI/arXiv-ID по строкам)."""

        n_lines = len([x for x in (inputs or "").splitlines() if x.strip()])
        query = f"Сравнение статей ({n_lines})" if n_lines else "Сравнение статей"
        job = Job(
            id=uuid.uuid4().hex[:12],
            owner_id=owner_id,
            query=query,
            kind="compare",
            inputs=inputs,
        )
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run_compare, job.id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_for_owner(self, owner_id: int) -> List[Job]:
        with self._lock:
            jobs = [j for j in self._jobs.values() if j.owner_id == owner_id]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def _set_progress(self, job: Job, label: str, fraction: float) -> None:
        """Колбэк прогресса из пайплайна (вызывается из рабочего потока)."""

        with self._lock:
            job.stage = label
            # Прогресс не должен уменьшаться и держится ниже 100% до статуса done.
            job.progress = max(job.progress, min(0.99, float(fraction)))
            stage, progress = job.stage, job.progress
        # Зеркалим прогресс в общую БД (вне лока), чтобы сайт мог показывать
        # прогресс и авто-обновлять список отчётов.
        self._mirror(job.id, stage=stage, progress=progress)

    def _mirror(self, job_id: str, **fields) -> None:
        """Best-effort зеркалирование состояния задачи в общую таблицу run_jobs."""

        try:
            from ..store import get_store

            get_store().update_run_job(job_id, **fields)
        except Exception:  # pragma: no cover - зеркало не должно ронять пайплайн
            logger.debug("run_jobs mirror failed for %s", job_id, exc_info=True)

    # ------------------------------------------------------------------ worker
    def _run(self, job_id: str) -> None:
        from ..pipeline.e2e import run_pipeline
        from ..store import get_store

        job = self.get(job_id)
        if job is None:
            return
        job.status = "running"
        job.progress = 0.02
        job.stage = "Запускаю обработку…"
        db = get_store()
        # Регистрируем запуск в общей БД (best-effort).
        try:
            db.create_run_job(job.id, job.owner_id, job.query, origin="web")
        except Exception:  # pragma: no cover
            logger.debug("create_run_job failed for %s", job.id, exc_info=True)
        settings = db.get_settings(job.owner_id)
        sources_csv = settings.sources_csv()
        sources = None if sources_csv == "all" else sources_csv.split(",")
        try:
            run_dir = run_pipeline(
                query=job.query,
                sources=sources,
                top_papers=settings.top_papers,
                search_limit=settings.search_limit,
                use_llm_for_hypotheses=True,
                generate_report=True,
                progress_fn=lambda label, frac: self._set_progress(job, label, frac),
            )
        except Exception as e:  # pragma: no cover - runtime path
            logger.exception("Web pipeline failed for %r", job.query)
            job.status = "error"
            job.message = f"{type(e).__name__}: {e}"
            self._mirror(job.id, status="error", stage=job.message)
            return

        report = Path(run_dir) / "report.html"
        if not report.exists():
            job.status = "error"
            job.message = "Отчёт не создан (возможно, ничего не найдено)."
            self._mirror(job.id, status="error", stage=job.message)
            return
        try:
            html = report.read_bytes()
            rid = db.add_report(job.owner_id, job.query, html, origin="web")
            job.report_id = rid
            job.status = "done"
            job.progress = 1.0
            job.stage = "Готово"
            job.message = "Готово."
            self._mirror(
                job.id, status="done", stage="Готово", progress=1.0, report_id=rid
            )
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to store web report for %r", job.query)
            job.status = "error"
            job.message = f"Не удалось сохранить отчёт: {e}"
            self._mirror(job.id, status="error", stage="Ошибка сохранения")

    # -------------------------------------------------------- compare worker
    def _run_compare(self, job_id: str) -> None:
        from ..pipeline.compare import CompareError, run_compare
        from ..store import get_store

        job = self.get(job_id)
        if job is None:
            return
        job.status = "running"
        job.progress = 0.02
        job.stage = "Запускаю сравнение…"
        db = get_store()
        try:
            db.create_run_job(job.id, job.owner_id, job.query, origin="compare")
        except Exception:  # pragma: no cover
            logger.debug("create_run_job failed for %s", job.id, exc_info=True)

        try:
            html = run_compare(
                job.inputs,
                progress_fn=lambda label, frac: self._set_progress(job, label, frac),
            )
        except CompareError as e:
            job.status = "error"
            job.message = str(e)
            self._mirror(job.id, status="error", stage=str(e))
            return
        except Exception as e:  # pragma: no cover - runtime path
            logger.exception("Compare failed for owner=%s", job.owner_id)
            job.status = "error"
            job.message = f"{type(e).__name__}: {e}"
            self._mirror(job.id, status="error", stage=job.message)
            return

        try:
            rid = db.add_report(job.owner_id, job.query, html, origin="compare")
            job.report_id = rid
            job.status = "done"
            job.progress = 1.0
            job.stage = "Готово"
            job.message = "Готово."
            self._mirror(
                job.id, status="done", stage="Готово", progress=1.0, report_id=rid
            )
        except Exception as e:  # pragma: no cover
            logger.exception("Failed to store compare report for owner=%s", job.owner_id)
            job.status = "error"
            job.message = f"Не удалось сохранить отчёт: {e}"
            self._mirror(job.id, status="error", stage="Ошибка сохранения")


_manager: Optional[JobManager] = None
_manager_lock = threading.Lock()


def get_job_manager() -> JobManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = JobManager()
    return _manager
