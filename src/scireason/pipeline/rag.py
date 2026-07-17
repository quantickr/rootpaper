# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Lightweight, dependency-free RAG for per-paper structured analysis.

This module provides an in-memory retriever over a single paper's text chunks.
It reuses :func:`scireason.llm.embed` (which has a deterministic hash-embedding
fallback, so it works fully offline) and a plain cosine similarity, avoiding any
requirement for a running Qdrant instance.

It also offers ``verify_quote`` for a lightweight hallucination check: a claim's
evidence quote is only trusted when it can be matched (fuzzily) against the
retrieved source text.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # embeddings are best-effort; hash fallback keeps this offline-safe
    from ..llm import embed
except Exception:  # pragma: no cover - defensive import
    embed = None  # type: ignore[assignment]


_WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


@dataclass
class Passage:
    """A retrievable unit of a paper (a chunk or an abstract slice)."""

    passage_id: str
    text: str
    page: Optional[int] = None
    source_kind: str = "text_chunk"


@dataclass
class RetrievedPassage:
    passage: Passage
    score: float


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        num += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return num / (math.sqrt(na) * math.sqrt(nb))


def _split_long_text(text: str, *, max_chars: int = 900) -> List[str]:
    """Split a long blob (e.g. an abstract) into paragraph-ish windows."""

    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts: List[str] = []
    buf: List[str] = []
    size = 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if size + len(sent) > max_chars and buf:
            parts.append(" ".join(buf))
            buf, size = [], 0
        buf.append(sent)
        size += len(sent)
    if buf:
        parts.append(" ".join(buf))
    return parts


def passages_from_record(record: Any, *, abstract: str = "") -> List[Passage]:
    """Build retrievable passages from a PaperRecord-like object.

    Uses ``evidence_units`` when a PDF was ingested; otherwise falls back to the
    record text / abstract split into windows. Works with duck typing so the RAG
    module has no hard dependency on the temporal KG builder types.
    """

    passages: List[Passage] = []
    units = list(getattr(record, "evidence_units", []) or [])
    for i, u in enumerate(units):
        txt = (getattr(u, "text", "") or "").strip()
        if len(txt) < 20:
            continue
        passages.append(
            Passage(
                passage_id=getattr(u, "chunk_id", "") or getattr(u, "unit_id", "") or f"u{i}",
                text=txt,
                page=getattr(u, "page", None),
                source_kind=getattr(u, "source_kind", "text_chunk"),
            )
        )

    if not passages:
        blob = (getattr(record, "text", "") or "").strip() or (abstract or "").strip()
        for i, window in enumerate(_split_long_text(blob)):
            passages.append(Passage(passage_id=f"abs{i}", text=window, source_kind="abstract"))
    elif abstract:
        # Also keep the abstract as a groundable source alongside chunks: LLM
        # quotes often paraphrase the abstract even when full text exists.
        for i, window in enumerate(_split_long_text(abstract)):
            passages.append(Passage(passage_id=f"abs{i}", text=window, source_kind="abstract"))

    return passages


class PaperRetriever:
    """In-memory dense retriever over a single paper's passages."""

    def __init__(self, passages: Sequence[Passage]) -> None:
        self.passages: List[Passage] = list(passages)
        self._vectors: List[List[float]] = []
        self._ready = False

    def _ensure_index(self) -> None:
        if self._ready:
            return
        self._ready = True
        if embed is None or not self.passages:
            self._vectors = []
            return
        try:
            self._vectors = embed([p.text for p in self.passages])
        except Exception:
            self._vectors = []

    def retrieve(self, query: str, *, limit: int = 5) -> List[RetrievedPassage]:
        self._ensure_index()
        if not self.passages:
            return []

        # Dense path (embeddings available).
        if self._vectors and embed is not None:
            try:
                qv = embed([query])[0]
                scored = [
                    RetrievedPassage(p, _cosine(qv, v))
                    for p, v in zip(self.passages, self._vectors)
                ]
                scored.sort(key=lambda r: r.score, reverse=True)
                return scored[:limit]
            except Exception:
                pass

        # Lexical fallback (no embeddings): token overlap.
        q_tokens = set(_WORD_RE.findall(query.lower()))
        scored = []
        for p in self.passages:
            toks = set(_WORD_RE.findall(p.text.lower()))
            overlap = len(q_tokens & toks) / float(max(1, len(q_tokens)))
            scored.append(RetrievedPassage(p, overlap))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:limit]

    def context_block(self, query: str, *, limit: int = 5, max_chars: int = 3500) -> Tuple[str, List[Passage]]:
        """Return a citation-tagged context string and the passages used.

        Each passage is prefixed with a ``[Sn]`` tag so the LLM can cite it and
        we can verify grounding afterwards.
        """

        hits = self.retrieve(query, limit=limit)
        lines: List[str] = []
        used: List[Passage] = []
        total = 0
        for i, h in enumerate(hits, 1):
            snippet = h.passage.text.strip()
            if total + len(snippet) > max_chars and used:
                break
            page = f" (стр. {h.passage.page})" if h.passage.page is not None else ""
            lines.append(f"[S{i}{page}] {snippet}")
            used.append(h.passage)
            total += len(snippet)
        return "\n\n".join(lines), used


def _normalize_for_match(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def verify_quote(quote: str, passages: Sequence[Passage], *, min_overlap: float = 0.6) -> Dict[str, Any]:
    """Lightweight hallucination check for a claim's evidence quote.

    Returns ``{"verified": bool, "overlap": float, "passage_id": str|None,
    "page": int|None}``. A quote is verified when a high fraction of its tokens
    appears (as a contiguous-ish bag) in some source passage.
    """

    q_tokens = _normalize_for_match(quote)
    if len(q_tokens) < 3:
        # Too short to verify meaningfully; treat as unverified but not a failure.
        return {"verified": False, "overlap": 0.0, "passage_id": None, "page": None}

    q_set = set(q_tokens)
    best = {"verified": False, "overlap": 0.0, "passage_id": None, "page": None}
    for p in passages:
        p_set = set(_normalize_for_match(p.text))
        if not p_set:
            continue
        overlap = len(q_set & p_set) / float(len(q_set))
        if overlap > best["overlap"]:
            best = {
                "verified": overlap >= min_overlap,
                "overlap": round(overlap, 3),
                "passage_id": p.passage_id,
                "page": p.page,
            }
    return best
