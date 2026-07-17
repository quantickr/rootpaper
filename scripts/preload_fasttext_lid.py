# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Предзагрузка офлайн-модели определения языка fastText (lid.176).

Модель ``lid.176.ftz`` (~1 МБ, 176 языков) точнее ``langdetect`` на коротких и
насыщенных терминами научных текстах. Скачивается на этапе сборки Docker-образа,
чтобы определение языка статьи работало офлайн.

Путь назначения: ``$FASTTEXT_LID_MODEL`` или ``<.cache>/fasttext/lid.176.ftz``.
Скрипт идемпотентен и НЕ падает при отсутствии сети (детекция деградирует до
langdetect / эвристики по алфавиту).
"""

from __future__ import annotations

import os
import sys
import urllib.request

URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"


def _target_path() -> str:
    env_path = os.environ.get("FASTTEXT_LID_MODEL")
    if env_path:
        return env_path
    base = os.path.join(os.getcwd(), ".cache", "fasttext")
    return os.path.join(base, "lid.176.ftz")


def main() -> int:
    dest = _target_path()
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[preload_fasttext] Модель уже на месте: {dest}")
        return 0

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        print(f"[preload_fasttext] Скачиваю lid.176.ftz -> {dest} ...")
        urllib.request.urlretrieve(URL, dest)
        size = os.path.getsize(dest)
        print(f"[preload_fasttext] Готово ({size} байт).")
    except Exception as exc:
        print(
            f"[preload_fasttext] Не удалось скачать модель (нет сети?): {exc}",
            file=sys.stderr,
        )
        # Не проваливаем сборку: детекция откатится на langdetect.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
