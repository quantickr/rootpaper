# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Предзагрузка офлайн-моделей перевода argostranslate.

Запускается на этапе сборки Docker-образа бота, чтобы перевод статей на русский
и русских запросов на английский работал без обращения в сеть в рантайме.

Статьи бывают не только на английском. argostranslate умеет переводить в
русский только напрямую из английского (пакет en->ru), поэтому для остальных
языков используется пивот ``xx -> en -> ru``. Чтобы это работало офлайн, образ
предзагружает:

* ``en->ru`` и ``ru->en`` — прямой перевод саммари и запросов;
* **все доступные пары ``xx->en``** из индекса argos — вход в пивот с любого
  поддерживаемого языка (es, de, fr, it, pt, zh, ja, uk, ...).

Скрипт идемпотентен (уже установленные пары пропускаются) и НЕ падает при
отсутствии сети — образ соберётся, но перевод деградирует до no-op.

Управление объёмом:
* ``ARGOS_PRELOAD=core``  — только en<->ru (лёгкий образ);
* ``ARGOS_PRELOAD=full``  — en<->ru + все xx->en (по умолчанию, максимум языков).
"""

from __future__ import annotations

import os
import sys

# Базовые пары для перевода саммари и запросов.
CORE_PAIRS = [("en", "ru"), ("ru", "en")]


def _target_pairs(available) -> list[tuple[str, str]]:
    """Список пар для установки в зависимости от режима ARGOS_PRELOAD."""

    mode = os.environ.get("ARGOS_PRELOAD", "full").strip().lower()
    pairs: list[tuple[str, str]] = list(CORE_PAIRS)
    if mode == "core":
        return pairs

    # full: добавляем все xx->en (вход в пивот xx->en->ru).
    seen = set(pairs)
    for pkg in available:
        pair = (pkg.from_code, pkg.to_code)
        if pkg.to_code == "en" and pkg.from_code != "en" and pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def main() -> int:
    try:
        import argostranslate.package as package
    except Exception as exc:  # pragma: no cover - argostranslate not installed
        print(f"[preload_argos] argostranslate недоступен: {exc}", file=sys.stderr)
        return 0

    try:
        package.update_package_index()
    except Exception as exc:
        print(
            f"[preload_argos] Не удалось обновить индекс пакетов (нет сети?): {exc}",
            file=sys.stderr,
        )

    try:
        available = package.get_available_packages()
    except Exception:
        available = []

    try:
        installed = {
            (p.from_code, p.to_code) for p in package.get_installed_packages()
        }
    except Exception:
        installed = set()

    wanted = _target_pairs(available)
    to_install = [pair for pair in wanted if pair not in installed]
    if not to_install:
        print("[preload_argos] Все требуемые языковые пары уже установлены.")
        return 0

    print(
        f"[preload_argos] К установке {len(to_install)} пар "
        f"(режим ARGOS_PRELOAD={os.environ.get('ARGOS_PRELOAD', 'full')})."
    )
    ok = 0
    failed = 0
    for from_code, to_code in to_install:
        try:
            package.install_package_for_language_pair(from_code, to_code)
            ok += 1
            print(f"[preload_argos] Готово: {from_code}->{to_code}")
        except Exception as exc:
            failed += 1
            print(
                f"[preload_argos] Не удалось установить {from_code}->{to_code}: {exc}",
                file=sys.stderr,
            )

    print(f"[preload_argos] Установлено {ok}, ошибок {failed}.")
    # Не проваливаем сборку, даже если что-то не скачалось: перевод опционален.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
