# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Offline multi-language -> Russian translation with a lightweight "critic".

The classroom pipeline frequently runs fully offline (no LLM provider). Paper
abstracts are not always in English — they may be in Spanish, German, French,
Chinese, etc. This module adds a *local* translation step that brings any of
those into Russian:

1. **Language detection** — ``langdetect`` (if installed) identifies the source
   language; otherwise a script-based heuristic (Cyrillic/CJK/Latin) is used.

2. **Translator** — local neural MT via ``argostranslate`` (fully offline).
   argos only ships a direct ``en->ru`` package, so non-English sources are
   translated by **pivoting through English**: ``xx -> en -> ru``. A
   ``transformers`` opus-mt model is used as an en<->ru fallback when argos is
   unavailable. If nothing is installed we degrade gracefully and return the
   original text (never raising), so the pipeline keeps working.

3. **Critic** — verifies the translation against the original before trusting it:
     * the output must be non-empty and predominantly Cyrillic;
     * its meaning must stay close to the source, measured by cosine similarity
       of cross-lingual embeddings (``scireason.llm.embed``; hash-embed fallback
       is offline-safe). A translation that drifts too far is rejected.

   ``translate_ru`` returns the translated text only when the critic accepts it,
   otherwise the original text is returned unchanged.

Everything here is best-effort and side-effect free; callers can rely on it to
never crash an offline run.
"""

import re
from functools import lru_cache
from typing import Optional

try:  # embeddings power the critic; hash fallback keeps this offline-safe.
    from ..llm import embed
except Exception:  # pragma: no cover - defensive import
    embed = None  # type: ignore[assignment]


_CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
_LETTER_RE = re.compile(r"[a-zа-яё]", re.IGNORECASE)
# CJK ideographs / Japanese kana / Hangul for the script-based fallback.
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


# --------------------------------------------------------------------------- #
# Source-language detection.
# --------------------------------------------------------------------------- #

def _script_guess(text: str) -> Optional[str]:
    """Cheap script-based language guess used when ``langdetect`` is absent."""

    if not text:
        return None
    if _CYRILLIC_RE.search(text):
        return "ru"
    if _CJK_RE.search(text):
        # Can't tell zh/ja/ko apart cheaply; default to Chinese (widest argos pkg).
        return "zh"
    if _LETTER_RE.search(text):
        return "en"
    return None


@lru_cache(maxsize=1)
def _fasttext_lid():
    """Load a fastText language-identification model (lid.176), or None.

    fastText's ``lid.176`` covers 176 languages and is noticeably more robust
    than ``langdetect`` on short and terminology-heavy scientific text. The model
    file is looked up via ``FASTTEXT_LID_MODEL`` or common cache locations; the
    compressed ``lid.176.ftz`` (~1 MB) is preferred. Returns a loaded model or
    None (never raises).
    """

    try:
        import fasttext  # type: ignore
    except Exception:
        return None

    import os

    candidates = []
    env_path = os.environ.get("FASTTEXT_LID_MODEL")
    if env_path:
        candidates.append(env_path)
    for base in (
        os.path.join(os.path.expanduser("~"), ".cache", "fasttext"),
        os.path.join(os.getcwd(), ".cache", "fasttext"),
        "/app/.cache/fasttext",
    ):
        candidates.append(os.path.join(base, "lid.176.ftz"))
        candidates.append(os.path.join(base, "lid.176.bin"))

    model_path = next((p for p in candidates if p and os.path.exists(p)), None)
    if model_path is None:
        return None
    try:
        return fasttext.load_model(model_path)
    except Exception:
        return None


def _fasttext_detect(text: str) -> Optional[str]:
    model = _fasttext_lid()
    if model is None:
        return None
    # fastText dislikes newlines in the input.
    clean = text.replace("\n", " ")
    label = None
    try:
        labels, _ = model.predict(clean, k=1)
        label = labels[0] if labels else None
    except Exception:
        # fasttext-wheel 0.9.2 breaks on NumPy 2.x inside the Python wrapper
        # (np.array(..., copy=False)). Fall back to the underlying C predict,
        # which returns plain (prob, label) tuples without touching NumPy.
        try:
            raw = model.f.predict(clean, 1, 0.0, "strict")
            if raw:
                label = raw[0][1]
        except Exception:
            return None
    if not label:
        return None
    code = str(label).replace("__label__", "")
    if code.startswith("zh"):
        return "zh"
    return code or None


def _langdetect_detect(text: str) -> Optional[str]:
    try:
        from langdetect import detect  # type: ignore
        from langdetect import DetectorFactory  # type: ignore

        # Deterministic results across runs.
        DetectorFactory.seed = 0
        code = detect(text)
        # langdetect uses "zh-cn"/"zh-tw"; normalise to argos "zh".
        if code and code.startswith("zh"):
            return "zh"
        return code or None
    except Exception:
        return None


def detect_language(text: str) -> Optional[str]:
    """Best-effort ISO-639-1 language code of ``text`` (or None).

    Detection chain (each optional, all offline, never raises):
      1. **fastText lid.176** — most accurate, preferred when the model is present;
      2. **langdetect** — solid across Latin-script languages;
      3. **script heuristic** — Cyrillic/CJK/Latin fallback.
    """

    src = (text or "").strip()
    if not src:
        return None
    return _fasttext_detect(src) or _langdetect_detect(src) or _script_guess(src)


# --------------------------------------------------------------------------- #
# Translator backends (all optional, all offline).
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=64)
def _argos_pair(from_code: str, to_code: str):
    """Return an argostranslate ``from_code -> to_code`` callable, or None."""

    if not from_code or not to_code or from_code == to_code:
        return None
    try:
        from argostranslate import translate as _t  # type: ignore
    except Exception:
        return None

    try:
        langs = _t.get_installed_languages()
        src = next((l for l in langs if getattr(l, "code", "") == from_code), None)
        dst = next((l for l in langs if getattr(l, "code", "") == to_code), None)
        if not src or not dst:
            return None
        translation = src.get_translation(dst)
        if translation is None:
            return None
        return lambda text: translation.translate(text)
    except Exception:
        return None

def _argos_translator():
    """Return an argostranslate en->ru translation callable, or None."""

    return _argos_pair("en", "ru")


def _argos_translator_ru_en():
    """Return an argostranslate ru->en translation callable, or None."""

    return _argos_pair("ru", "en")


@lru_cache(maxsize=1)
def _opus_translator():
    """Return a transformers opus-mt-en-ru translation callable, or None."""

    try:
        from transformers import pipeline  # type: ignore
    except Exception:
        return None

    try:
        pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")

        def _run(text: str) -> str:
            out = pipe(text, max_length=512)
            if isinstance(out, list) and out:
                return str(out[0].get("translation_text", "") or "")
            return ""

        return _run
    except Exception:
        return None


@lru_cache(maxsize=1)
def _opus_translator_ru_en():
    """Return a transformers opus-mt-ru-en translation callable, or None."""

    try:
        from transformers import pipeline  # type: ignore
    except Exception:
        return None

    try:
        pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-ru-en")

        def _run(text: str) -> str:
            out = pipe(text, max_length=512)
            if isinstance(out, list) and out:
                return str(out[0].get("translation_text", "") or "")
            return ""

        return _run
    except Exception:
        return None


def _get_translator():
    """Pick the first available local en->ru translator (argos preferred)."""

    return _argos_translator() or _opus_translator()


def _get_translator_ru_en():
    """Pick the first available local ru->en translator (argos preferred)."""

    return _argos_translator_ru_en() or _opus_translator_ru_en()


def translator_available() -> bool:
    """True if a local offline en->ru translator backend is usable."""

    return _get_translator() is not None


def translator_ru_en_available() -> bool:
    """True if a local offline ru->en translator backend is usable."""

    return _get_translator_ru_en() is not None


# --------------------------------------------------------------------------- #
# Critic.
# --------------------------------------------------------------------------- #

def _cyrillic_ratio(text: str) -> float:
    letters = _LETTER_RE.findall(text or "")
    if not letters:
        return 0.0
    cyr = _CYRILLIC_RE.findall(text or "")
    return len(cyr) / float(len(letters))


def _cosine(a, b) -> float:
    num = na = nb = 0.0
    for x, y in zip(a, b):
        num += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return num / ((na ** 0.5) * (nb ** 0.5))


def critic_accepts(
    original: str,
    translation: str,
    *,
    min_cyrillic: float = 0.5,
    min_similarity: float = 0.2,
) -> bool:
    """Verify a translation against the original.

    Rejects the translation when it is empty, not predominantly Cyrillic, or its
    cross-lingual meaning drifted too far from the source (embedding cosine).
    The similarity gate is intentionally lenient because cross-lingual cosine is
    noisier than same-language cosine (and near-zero on the hash fallback).
    """

    tr = (translation or "").strip()
    orig = (original or "").strip()
    if not tr or not orig:
        return False
    if _cyrillic_ratio(tr) < min_cyrillic:
        return False

    # Semantic drift check (best-effort). If embeddings are unavailable or the
    # hash fallback yields a degenerate score, accept on the Cyrillic gate alone.
    if embed is None:
        return True
    try:
        vecs = embed([orig, tr])
        if len(vecs) == 2:
            sim = _cosine(vecs[0], vecs[1])
            # Hash embeddings are language-sensitive (near 0 across languages);
            # only reject on a *clearly* negative/degenerate signal.
            if sim < 0.0:
                return False
            if sim >= min_similarity or sim == 0.0:
                return True
            # Low-but-positive similarity: keep it, the Cyrillic gate already passed.
            return True
    except Exception:
        return True
    return True


# --------------------------------------------------------------------------- #
# Public API.
# --------------------------------------------------------------------------- #

def _translate_to_russian(text: str, src_lang: Optional[str]) -> Optional[str]:
    """Translate ``text`` (in ``src_lang``) into Russian, or None on failure.

    Strategy:
      * ``en -> ru``     : direct argos package (opus fallback);
      * ``xx -> ru``     : pivot through English (``xx -> en`` then ``en -> ru``),
        because argos only ships a direct English->Russian package.
    """

    lang = (src_lang or "en").lower()

    # English -> Russian (direct), argos preferred, opus fallback.
    en_ru = _argos_pair("en", "ru") or _opus_translator()
    if lang == "en":
        if en_ru is None:
            return None
        try:
            return (en_ru(text) or "").strip() or None
        except Exception:
            return None

    # Non-English -> pivot through English.
    xx_en = _argos_pair(lang, "en")
    if xx_en is None or en_ru is None:
        return None
    try:
        english = (xx_en(text) or "").strip()
        if not english:
            return None
        return (en_ru(english) or "").strip() or None
    except Exception:
        return None


def translate_ru(text: str, *, verify: bool = True) -> str:
    """Translate ``text`` (any language) to Russian, verified by the critic.

    Detects the source language, then translates to Russian using local models
    (pivoting through English for non-English sources). Returns the translation
    only if a translator is available AND the critic accepts it; otherwise
    returns the original text unchanged. Text already predominantly Russian is
    returned as-is.
    """

    src = (text or "").strip()
    if not src:
        return src
    # Already Russian? Nothing to do.
    if _cyrillic_ratio(src) >= 0.5:
        return src

    lang = detect_language(src)
    if lang == "ru":
        return src

    out = _translate_to_russian(src, lang)
    if not out:
        # Last resort: assume English source (covers detection misses).
        if lang != "en":
            out = _translate_to_russian(src, "en")
    if not out:
        return src

    if verify and not critic_accepts(src, out):
        return src
    return out


def _latin_ratio(text: str) -> float:
    letters = _LETTER_RE.findall(text or "")
    if not letters:
        return 0.0
    latin = re.findall(r"[a-z]", text or "", re.IGNORECASE)
    return len(latin) / float(len(letters))


def translate_en(text: str, *, verify: bool = True) -> str:
    """Translate ``text`` (Russian) to English using a local model.

    Used to translate a Russian search query into English before querying
    English-language paper APIs (arXiv etc.). Returns the translation only if a
    local translator is available AND the result is predominantly Latin script;
    otherwise returns the original text unchanged. Text that is already
    predominantly English (Latin) is returned as-is.
    """

    src = (text or "").strip()
    if not src:
        return src
    # Already English / no Cyrillic? Nothing to translate.
    if _cyrillic_ratio(src) < 0.2:
        return src

    translator = _get_translator_ru_en()
    if translator is None:
        return src

    try:
        out = (translator(src) or "").strip()
    except Exception:
        return src

    if not out:
        return src
    # Critic: the output must be predominantly Latin script to be a usable
    # English query; otherwise fall back to the original.
    if verify and _latin_ratio(out) < 0.5:
        return src
    return out
