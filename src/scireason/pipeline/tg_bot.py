# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Telegram bot front-end for the scireason pipeline.

A student sends a topic **in Russian**; the bot runs the full end-to-end
pipeline (query -> arXiv & other APIs -> temporal KG -> interactive report)
and replies with the self-contained ``report.html`` file.

Design
------
* Built on **aiogram 3.x** (async).
* The pipeline (:func:`scireason.pipeline.e2e.run_pipeline`) is synchronous and
  long-running (network I/O, PDF download, KG build). We therefore:
    - accept requests into an ``asyncio.Queue``;
    - process them with a small pool of worker tasks (bounded concurrency), so
      several users can be served without one blocking the others;
    - run each blocking pipeline call in a thread executor so the event loop
      stays responsive (keepalive typing, other chats).
* The Russian query is translated to English *inside* the pipeline for the
  API search, while the report keeps the original Russian query.

Per-user state
--------------
* **Настройки** (источники, ``search_limit``, ``top_papers``) хранятся в SQLite
  (:mod:`scireason.pipeline.tg_store`) и применяются ко всем запросам
  пользователя, пока он не изменит их через ``/settings``.
* **История отчётов** (тема, дата, сам HTML) тоже в SQLite — команда ``/story``
  показывает прошлые отчёты по 5 штук с пагинацией.

Token is read from ``settings.telegram_bot_token`` (i.e. ``TELEGRAM_BOT_TOKEN``
in ``.env``) and can be overridden via :func:`run_bot`.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..config import settings
from .e2e import run_pipeline
from .tg_store import ALL_SOURCES, TgStore, UserSettings

logger = logging.getLogger(__name__)

# Telegram rejects documents larger than 50 MB via the Bot API.
_MAX_TG_DOCUMENT_BYTES = 50 * 1024 * 1024

# Сколько отчётов показывать за одну страницу /story.
_STORY_PAGE_SIZE = 5

# Человекочитаемые названия источников для клавиатуры настроек.
_SOURCE_TITLES: Dict[str, str] = {
    "openalex": "OpenAlex",
    "semantic_scholar": "Semantic Scholar",
    "crossref": "Crossref",
    "arxiv": "arXiv",
    "pubmed": "PubMed",
    "europe_pmc": "Europe PMC",
    "biorxiv": "bioRxiv",
}


@dataclass
class _Job:
    chat_id: int
    user_id: int
    status_message_id: int
    query: str


def _find_report(run_dir: Path) -> Optional[Path]:
    report = run_dir / "report.html"
    return report if report.exists() else None


def _run_pipeline_blocking(
    query: str,
    *,
    sources: Optional[List[str]],
    top_papers: int,
    search_limit: int,
) -> Path:
    """Blocking pipeline call (executed in a thread executor)."""

    return run_pipeline(
        query=query,
        sources=sources,
        top_papers=top_papers,
        search_limit=search_limit,
        # Keep the offline-safe defaults; the report + Russian translation work
        # without an LLM provider.
        use_llm_for_hypotheses=True,
        generate_report=True,
    )


def build_bot_app(
    token: str,
    *,
    workers: int = 2,
    top_papers: int = 10,
    search_limit: int = 30,
    store: Optional[TgStore] = None,
):
    """Assemble the aiogram Bot + Dispatcher with a queued worker pool.

    Imports of aiogram are local so the rest of the package (and its tests) do
    not require aiogram to be installed.
    """

    from aiogram import Bot, Dispatcher, F
    from aiogram.filters import Command, CommandStart
    from aiogram.types import (
        CallbackQuery,
        FSInputFile,
        InlineKeyboardButton,
        InlineKeyboardMarkup,
        Message,
    )

    bot = Bot(token=token)
    dp = Dispatcher()

    db = store if store is not None else TgStore()

    queue: "asyncio.Queue[_Job]" = asyncio.Queue()

    # Пользователи, от которых бот сейчас ждёт число вместо поискового запроса.
    # user_id -> "search_limit" | "top_papers".
    awaiting_number: Dict[int, str] = {}

    # ------------------------------------------------------------------ helpers
    def _settings_keyboard(s: UserSettings) -> "InlineKeyboardMarkup":
        rows: List[List[InlineKeyboardButton]] = []
        for src in ALL_SOURCES:
            mark = "✅" if src in s.sources else "⬜"
            title = _SOURCE_TITLES.get(src, src)
            rows.append(
                [
                    InlineKeyboardButton(
                        text=f"{mark} {title}",
                        callback_data=f"src:{src}",
                    )
                ]
            )
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"🔢 search-limit: {s.search_limit}",
                    callback_data="num:search_limit",
                )
            ]
        )
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"📄 top-papers: {s.top_papers}",
                    callback_data="num:top_papers",
                )
            ]
        )
        return InlineKeyboardMarkup(inline_keyboard=rows)

    def _settings_text(s: UserSettings) -> str:
        chosen = ", ".join(_SOURCE_TITLES.get(x, x) for x in s.sources) or "— (будут все)"
        return (
            "<b>Твои настройки</b>\n\n"
            f"Источники: {chosen}\n"
            f"search-limit (сколько статей искать): <b>{s.search_limit}</b>\n"
            f"top-papers (сколько взять в отчёт): <b>{s.top_papers}</b>\n\n"
            "Нажимай на источники, чтобы включать/выключать их. "
            "Кнопки с числами попросят ввести новое значение сообщением."
        )

    def _story_page_kb(user_id: int, offset: int) -> "Optional[InlineKeyboardMarkup]":
        total = db.count_reports(user_id)
        rows: List[List[InlineKeyboardButton]] = []
        page = db.list_reports(user_id, offset=offset, limit=_STORY_PAGE_SIZE)
        for r in page:
            when = datetime.fromtimestamp(r.created_at).strftime("%Y-%m-%d %H:%M")
            title = r.query if len(r.query) <= 40 else r.query[:39] + "…"
            rows.append(
                [
                    InlineKeyboardButton(
                        text=f"⬇ {when} · {title}",
                        callback_data=f"dl:{r.id}",
                    )
                ]
            )
        nav: List[InlineKeyboardButton] = []
        if offset > 0:
            prev_off = max(0, offset - _STORY_PAGE_SIZE)
            nav.append(
                InlineKeyboardButton(text="◀ Назад", callback_data=f"story:{prev_off}")
            )
        if offset + _STORY_PAGE_SIZE < total:
            nav.append(
                InlineKeyboardButton(
                    text="Ещё 5 ▶", callback_data=f"story:{offset + _STORY_PAGE_SIZE}"
                )
            )
        if nav:
            rows.append(nav)
        if not rows:
            return None
        return InlineKeyboardMarkup(inline_keyboard=rows)

    # ----------------------------------------------------------------- commands
    @dp.message(CommandStart())
    async def _start(message: Message) -> None:
        await message.answer(
            "Привет! Пришли тему научного поиска на русском — я найду статьи "
            "(в т.ч. с arXiv), построю интерактивный граф и русские саммари, "
            "и верну HTML-отчёт.\n\n"
            "Команды:\n"
            "• /settings — источники и лимиты под себя\n"
            "• /story — прошлые отчёты (по 5 штук)\n"
            "• /help — подробности\n\n"
            "Есть и веб-версия: там можно завести аккаунт по почте и привязать "
            "этот Telegram — история отчётов объединится.\n\n"
            "Например: <b>прогнозирование трафика с помощью глубокого обучения</b>",
            parse_mode="HTML",
        )

    @dp.message(Command("help"))
    async def _help(message: Message) -> None:
        await message.answer(
            "Просто отправь тему на русском (или английском). "
            "Обработка занимает время (поиск, скачивание PDF, построение графа), "
            "я пришлю отчёт файлом, как только он будет готов.\n\n"
            "• /settings — выбрать источники и задать search-limit / top-papers. "
            "Настройки запоминаются и применяются ко всем твоим запросам.\n"
            "• /story — показать прошлые отчёты по 5 штук; жми на запись, "
            "чтобы скачать HTML."
        )

    @dp.message(Command("settings"))
    async def _settings_cmd(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else message.chat.id
        awaiting_number.pop(user_id, None)
        s = db.get_settings(user_id)
        await message.answer(
            _settings_text(s), parse_mode="HTML", reply_markup=_settings_keyboard(s)
        )

    @dp.message(Command("story"))
    async def _story_cmd(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else message.chat.id
        total = db.count_reports(user_id)
        if total == 0:
            await message.answer(
                "У тебя пока нет сохранённых отчётов. Пришли тему — и первый появится."
            )
            return
        kb = _story_page_kb(user_id, 0)
        await message.answer(
            f"Твои отчёты (всего {total}). Жми на запись, чтобы скачать HTML:",
            reply_markup=kb,
        )

    # ------------------------------------------------------------- callbacks
    @dp.callback_query(F.data.startswith("src:"))
    async def _cb_toggle_source(cb: CallbackQuery) -> None:
        user_id = cb.from_user.id
        src = cb.data.split(":", 1)[1]
        s = db.toggle_source(user_id, src)
        try:
            await cb.message.edit_text(
                _settings_text(s),
                parse_mode="HTML",
                reply_markup=_settings_keyboard(s),
            )
        except Exception:  # pragma: no cover - "message is not modified" etc.
            pass
        await cb.answer()

    @dp.callback_query(F.data.startswith("num:"))
    async def _cb_ask_number(cb: CallbackQuery) -> None:
        user_id = cb.from_user.id
        field = cb.data.split(":", 1)[1]
        awaiting_number[user_id] = field
        pretty = "search-limit" if field == "search_limit" else "top-papers"
        await cb.message.answer(
            f"Пришли новое число для <b>{pretty}</b> одним сообщением (например, 20).",
            parse_mode="HTML",
        )
        await cb.answer()

    @dp.callback_query(F.data.startswith("story:"))
    async def _cb_story_page(cb: CallbackQuery) -> None:
        user_id = cb.from_user.id
        try:
            offset = int(cb.data.split(":", 1)[1])
        except ValueError:
            offset = 0
        kb = _story_page_kb(user_id, offset)
        total = db.count_reports(user_id)
        try:
            await cb.message.edit_text(
                f"Твои отчёты (всего {total}). Жми на запись, чтобы скачать HTML:",
                reply_markup=kb,
            )
        except Exception:  # pragma: no cover
            pass
        await cb.answer()

    @dp.callback_query(F.data.startswith("dl:"))
    async def _cb_download(cb: CallbackQuery) -> None:
        from aiogram.types import BufferedInputFile

        user_id = cb.from_user.id
        try:
            report_id = int(cb.data.split(":", 1)[1])
        except ValueError:
            await cb.answer("Некорректная запись.", show_alert=True)
            return
        html = db.get_report_html(user_id, report_id)
        if html is None:
            await cb.answer("Отчёт не найден.", show_alert=True)
            return
        await cb.message.answer_document(
            BufferedInputFile(html, filename=f"report_{report_id}.html"),
            caption="Прошлый отчёт. Открой в браузере — внутри граф и русские саммари.",
        )
        await cb.answer()

    # --------------------------------------------------------------- text query
    @dp.message(F.text & ~F.text.startswith("/"))
    async def _on_query(message: Message) -> None:
        user_id = message.from_user.id if message.from_user else message.chat.id
        # Сохраняем username, чтобы упростить объединение аккаунтов на сайте.
        if message.from_user is not None:
            try:
                db.ensure_tg_user(user_id, message.from_user.username)
            except Exception:  # pragma: no cover - best-effort
                pass
        text = (message.text or "").strip()

        # Если ждём число для настройки — интерпретируем сообщение как число.
        field = awaiting_number.get(user_id)
        if field is not None:
            try:
                value = int(text)
                if value < 1:
                    raise ValueError
            except ValueError:
                await message.answer("Нужно целое число ≥ 1. Попробуй ещё раз.")
                return
            if field == "search_limit":
                s = db.set_search_limit(user_id, value)
            else:
                s = db.set_top_papers(user_id, value)
            awaiting_number.pop(user_id, None)
            await message.answer(
                "Сохранил.", reply_markup=None
            )
            await message.answer(
                _settings_text(s),
                parse_mode="HTML",
                reply_markup=_settings_keyboard(s),
            )
            return

        if not text:
            await message.answer("Пустой запрос. Пришли тему текстом.")
            return
        status = await message.answer(
            "Принял запрос в очередь. Обрабатываю — это может занять несколько минут…"
        )
        await queue.put(
            _Job(
                chat_id=message.chat.id,
                user_id=user_id,
                status_message_id=status.message_id,
                query=text,
            )
        )

    # ------------------------------------------------------------- job pipeline
    async def _process_job(job: _Job) -> None:
        from aiogram.types import FSInputFile

        loop = asyncio.get_running_loop()
        user = db.get_settings(job.user_id)
        sources_csv = user.sources_csv()
        sources_list = None if sources_csv == "all" else sources_csv.split(",")
        try:
            run_dir = await loop.run_in_executor(
                None,
                lambda: _run_pipeline_blocking(
                    job.query,
                    sources=sources_list,
                    top_papers=user.top_papers,
                    search_limit=user.search_limit,
                ),
            )
        except Exception as e:  # pragma: no cover - runtime failure path
            logger.exception("Pipeline failed for query=%r", job.query)
            await bot.send_message(
                job.chat_id,
                f"Не удалось обработать запрос: {type(e).__name__}: {e}",
            )
            return

        report = _find_report(Path(run_dir))
        if report is None:
            await bot.send_message(
                job.chat_id,
                "Пайплайн завершился, но HTML-отчёт не был создан "
                "(возможно, ничего не нашлось). Попробуй переформулировать тему.",
            )
            return

        # Сохраняем отчёт в историю (BLOB), чтобы /story работал даже без runs/.
        # Заодно готовим публичную ссылку на HTML-версию на сайте (по токену).
        share_url: Optional[str] = None
        try:
            html_bytes = report.read_bytes()
            report_id = db.add_report(job.user_id, job.query, html_bytes)
            try:
                token = db.create_report_share(report_id)
                base = (settings.web_base_url or "").rstrip("/")
                if base:
                    share_url = f"{base}/r/{token}"
            except Exception:  # pragma: no cover - ссылка best-effort
                logger.exception("Failed to create share link for %r", job.query)
        except Exception:  # pragma: no cover - history is best-effort
            logger.exception("Failed to store report in history for %r", job.query)

        link_line = f"\n\n🔗 Открыть на сайте: {share_url}" if share_url else ""

        size = report.stat().st_size
        if size > _MAX_TG_DOCUMENT_BYTES:
            too_big = (
                f"Отчёт готов, но он слишком большой для Telegram "
                f"({size // (1024 * 1024)} МБ)."
            )
            if share_url:
                too_big += f"\n\n🔗 Открой его на сайте: {share_url}"
            else:
                too_big += f" Он сохранён локально: {report}"
            await bot.send_message(job.chat_id, too_big)
            return

        try:
            await bot.send_document(
                job.chat_id,
                FSInputFile(str(report), filename="report.html"),
                caption=(
                    f"Готово: «{job.query}». Открой файл в браузере — "
                    f"внутри интерактивный граф и русские саммари.{link_line}"
                ),
            )
        except Exception as e:  # pragma: no cover - runtime failure path
            logger.exception("Failed to send report for query=%r", job.query)
            await bot.send_message(
                job.chat_id,
                f"Отчёт готов ({report}), но не удалось отправить файл: {type(e).__name__}: {e}",
            )

    async def _worker(worker_id: int) -> None:
        while True:
            job = await queue.get()
            try:
                await _process_job(job)
            finally:
                queue.task_done()

    async def _on_startup() -> None:
        for i in range(max(1, workers)):
            asyncio.create_task(_worker(i))
        logger.info("Started %d pipeline worker(s)", workers)

    dp.startup.register(_on_startup)
    return bot, dp


def run_bot(
    *,
    token: Optional[str] = None,
    workers: int = 2,
    top_papers: int = 10,
    search_limit: int = 30,
) -> None:
    """Start long-polling the Telegram bot (blocking).

    ``token`` falls back to ``settings.telegram_bot_token`` (``.env``:
    ``TELEGRAM_BOT_TOKEN``). Raises ``RuntimeError`` if no token is configured.
    """

    tok = (token or settings.telegram_bot_token or "").strip()
    if not tok:
        raise RuntimeError(
            "Не задан токен Telegram-бота. Укажи TELEGRAM_BOT_TOKEN в .env "
            "или передай --token."
        )

    logging.basicConfig(level=logging.INFO)
    bot, dp = build_bot_app(
        tok, workers=workers, top_papers=top_papers, search_limit=search_limit
    )

    async def _main() -> None:
        await dp.start_polling(bot)

    asyncio.run(_main())
