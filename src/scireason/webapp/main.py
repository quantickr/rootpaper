# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Точка входа для запуска веб-сайта через uvicorn."""

import logging

from ..config import settings


def main() -> None:
    try:
        import uvicorn
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Web extra is not installed. Install with: pip install -e '.[web]'"
        ) from e

    logging.basicConfig(level=logging.INFO)
    from .app import create_app

    uvicorn.run(
        create_app(),
        host=settings.web_host,
        port=int(settings.web_port),
    )


if __name__ == "__main__":
    main()
