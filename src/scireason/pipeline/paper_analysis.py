# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Structured scientific analysis of a paper (in Russian) for students.

For each paper this produces:

* a structured breakdown: hypothesis, method, data, results, limitations;
* a suggested research plan the student could carry out;
* a concept map of *phrase-level* concepts and named relations between them
  (not just single words);
* citation grounding: each structured field carries an evidence quote which is
  verified (lexically) against retrieved source passages to flag possible
  hallucinations.

All LLM calls go through :func:`scireason.llm.chat_json` and degrade gracefully
to a deterministic, abstract-based fallback when the LLM is unavailable (mock /
offline), so classroom runs never break.
"""

import re
from typing import Any, Dict, List, Optional, Sequence

from .rag import PaperRetriever, Passage, passages_from_record, verify_quote

try:
    from ..llm import chat_json
except Exception:  # pragma: no cover - defensive import
    chat_json = None  # type: ignore[assignment]

try:  # local offline EN->RU translator (+ critic); no-op if unavailable.
    from .translate import translate_ru
except Exception:  # pragma: no cover - defensive import
    translate_ru = None  # type: ignore[assignment]


def _to_russian(text: str) -> str:
    """Best-effort local translation to Russian; returns input unchanged on failure."""

    if translate_ru is None or not (text or "").strip():
        return text
    try:
        return translate_ru(text)
    except Exception:
        return text


# --- The fields we extract, with Russian display labels for the report. ---
STRUCTURE_FIELDS: List[tuple[str, str]] = [
    ("hypothesis", "Гипотеза"),
    ("method", "Метод"),
    ("data", "Данные"),
    ("results", "Результаты"),
    ("limitations", "Ограничения"),
]

_STRUCTURE_SCHEMA = (
    "JSON-объект на РУССКОМ языке с полями: "
    '{"hypothesis": {"text": str, "quote": str}, '
    '"method": {"text": str, "quote": str}, '
    '"data": {"text": str, "quote": str}, '
    '"results": {"text": str, "quote": str}, '
    '"limitations": {"text": str, "quote": str}, '
    '"research_plan": [str, str, str]}. '
    "Поле text — краткое объяснение на русском (1-2 предложения). "
    "Поле quote — ДОСЛОВНАЯ цитата из предоставленного контекста (на языке оригинала), "
    "подтверждающая утверждение; если подтверждения нет — пустая строка. "
    "research_plan — 3-5 конкретных шагов собственного исследования школьника."
)

_CONCEPT_SCHEMA = (
    "JSON-объект: "
    '{"concepts": [str, ...], "relations": [{"source": str, "relation": str, "target": str}, ...]}. '
    "concepts — 6-14 КЛЮЧЕВЫХ ПОНЯТИЙ статьи в виде осмысленных СЛОВОСОЧЕТАНИЙ/логических выражений "
    "(например 'твёрдоэлектролитная граница раздела', 'рост дендритов лития'), НЕ отдельных слов. "
    "relations — связи между этими понятиями; relation — короткая осмысленная фраза "
    "('приводит к', 'снижает', 'зависит от', 'измеряется через'). "
    "source и target ДОЛЖНЫ быть из списка concepts."
)

_WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip() + "\u2026"


def _fallback_structure(title: str, abstract: str) -> Dict[str, Any]:
    """Deterministic best-effort structure when the LLM is unavailable."""

    sents = [s.strip() for s in _SENT_RE.split(abstract or "") if s.strip()]

    def pick(keywords: Sequence[str], default_idx: int) -> str:
        """Pick the best matching source sentence and translate it to Russian.

        The chosen sentence is used as the human-readable ``text``; it is
        translated locally so the offline breakdown reads in Russian. The
        original sentence is preserved separately as the evidence ``quote`` so
        citation grounding still matches the source verbatim.
        """

        for s in sents:
            low = s.lower()
            if any(k in low for k in keywords):
                return _truncate(s, 300)
        if sents and 0 <= default_idx < len(sents):
            return _truncate(sents[default_idx], 300)
        return "н/д"

    def field(keywords: Sequence[str], default_idx: int) -> Dict[str, str]:
        src = pick(keywords, default_idx)
        if src == "н/д":
            return {"text": "н/д", "quote": ""}
        ru = _to_russian(src)
        # If translation happened, keep the original as the groundable quote.
        quote = src if ru.strip() != src.strip() else ""
        return {"text": ru, "quote": quote}

    return {
        "hypothesis": field(["hypothes", "propose", "гипотез", "предполаг"], 0),
        "method": field(["method", "approach", "метод", "использ"], 1),
        "data": field(["dataset", "data", "sample", "данн", "выборк"], 2),
        "results": field(["result", "show", "find", "результат", "показал"], -1 if not sents else len(sents) - 1),
        "limitations": field(["limitation", "however", "ограничен", "однако"], 0),
        "research_plan": [
            "Сформулируйте свой исследовательский вопрос на основе гипотезы статьи.",
            "Определите, какие данные/материалы вам нужны и как их получить.",
            "Спланируйте эксперимент или анализ, повторяющий или проверяющий ключевой результат.",
        ],
    }


def analyze_structure(
    *,
    title: str,
    abstract: str,
    abstract_ru: Optional[str] = None,
    retriever: PaperRetriever,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Extract the structured breakdown + research plan (Russian), grounded via RAG.

    ``abstract`` — оригинальный текст; ``abstract_ru`` принимается для единства
    сигнатуры с ``extract_concept_map``, но здесь не используется: фолбэк
    структуры сам переводит выбранное предложение и сохраняет оригинал как
    дословную цитату для заземления.
    """

    if not use_llm or chat_json is None:
        data = _fallback_structure(title, abstract)
        return _ground(data, retriever, query=f"{title} {abstract}")

    # Retrieve grounding context for the structural questions.
    query = (
        f"{title}. Гипотеза, метод, данные, результаты и ограничения исследования."
    )
    context, passages = retriever.context_block(query, limit=6)
    if not context:
        context = _truncate(abstract or title, 3000)

    system = (
        "Ты — научный ассистент, помогающий школьнику разобрать научную статью. "
        "Отвечай на РУССКОМ языке, кратко и по делу. Опирайся ТОЛЬКО на приведённый контекст. "
        "Если сведений нет — пиши 'н/д'. Не выдумывай факты."
    )
    user = f"Название: {title}\n\nКонтекст (фрагменты статьи):\n{context}"
    try:
        data = chat_json(system, user, _STRUCTURE_SCHEMA, temperature=0.2)
        if not isinstance(data, dict) or not _has_structure_content(data):
            # Empty/degenerate response (e.g. the offline mock returns {}): use the
            # deterministic abstract-based structure instead of showing "н/д".
            raise ValueError("empty or invalid structure")
    except Exception:
        data = _fallback_structure(title, abstract)

    return _ground(data, retriever, query=query, passages=passages)


def _has_structure_content(data: Dict[str, Any]) -> bool:
    """True if at least one structured field carries real text (not empty/'н/д')."""

    for key, _label in STRUCTURE_FIELDS:
        field = data.get(key)
        if isinstance(field, str):
            text = field.strip()
        elif isinstance(field, dict):
            text = str(field.get("text", "") or "").strip()
        else:
            text = ""
        if text and text.lower() not in {"н/д", "n/a", "нет данных"}:
            return True
    return False


def _ground(
    data: Dict[str, Any],
    retriever: PaperRetriever,
    *,
    query: str,
    passages: Optional[Sequence[Passage]] = None,
) -> Dict[str, Any]:
    """Attach citation verification to each structured field.

    Quotes are verified against ALL of the paper's passages (not just the
    retrieved context slice) so a valid quote from any part of the paper counts.
    """

    passages = list(retriever.passages) or list(passages or [])

    out: Dict[str, Any] = {}
    for key, _label in STRUCTURE_FIELDS:
        field = data.get(key) or {}
        if isinstance(field, str):
            field = {"text": field, "quote": ""}
        text = str(field.get("text", "") or "").strip() or "н/д"
        quote = str(field.get("quote", "") or "").strip()
        verification = verify_quote(quote, passages) if quote else {
            "verified": False, "overlap": 0.0, "passage_id": None, "page": None,
        }
        out[key] = {
            "text": text,
            "quote": quote,
            "verified": bool(verification["verified"]),
            "page": verification.get("page"),
        }

    plan = data.get("research_plan") or []
    if isinstance(plan, str):
        plan = [plan]
    out["research_plan"] = [str(x).strip() for x in plan if str(x).strip()][:6]
    return out


def _fallback_concept_map(title: str, abstract: str) -> Dict[str, Any]:
    """Extract multi-word concepts heuristically when the LLM is unavailable."""

    text = f"{title}. {abstract}".lower()
    # Кандидаты — последовательности из 2-4 слов. Токен ``[^\W_]+`` покрывает
    # буквы/цифры ЛЮБОГО алфавита (без подчёркивания), поэтому диакритика
    # (tr/cs/pt) и кириллица не режутся на осколки.
    token = r"[^\W_]+"
    phrases = re.findall(rf"{token}(?:\s+{token}){{1,3}}", text, re.UNICODE)
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was",
        "были", "быть", "этот", "как", "или", "что", "это", "для", "при",
        "над", "под", "без", "его", "она", "они", "также", "который",
    }
    freq: Dict[str, int] = {}
    for ph in phrases:
        toks = [t for t in ph.split() if len(t) > 3 and t not in stop]
        if len(toks) < 2:
            continue
        key = " ".join(toks[:4])
        freq[key] = freq.get(key, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    concepts = [c for c, _ in ranked[:10]]
    relations = []
    for i in range(len(concepts) - 1):
        relations.append({"source": concepts[i], "relation": "связано с", "target": concepts[i + 1]})
    return {"concepts": concepts, "relations": relations[:12]}


def extract_concept_map(
    *,
    title: str,
    abstract: str,
    abstract_ru: Optional[str] = None,
    retriever: PaperRetriever,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Extract a phrase-level concept map (concepts + named relations).

    ``abstract`` — оригинал (для RAG-контекста LLM); ``abstract_ru`` — русская
    версия, из которой строится офлайн-фолбэк карты понятий (иначе узлы графа
    остаются на языке оригинала и режутся на осколки на диакритике).
    """

    ru = abstract_ru if abstract_ru is not None else abstract

    if not use_llm or chat_json is None:
        return _fallback_concept_map(title, ru)

    context, _ = retriever.context_block(f"{title}. Ключевые понятия и связи.", limit=5)
    if not context:
        context = _truncate(abstract or title, 3000)

    system = (
        "Ты строишь карту понятий научной статьи для школьника. "
        "Понятия — это осмысленные СЛОВОСОЧЕТАНИЯ (логические выражения), а не отдельные слова. "
        "Связи должны быть содержательными. Опирайся только на контекст."
    )
    user = f"Название: {title}\n\nКонтекст:\n{context}"
    try:
        data = chat_json(system, user, _CONCEPT_SCHEMA, temperature=0.2)
        concepts = [str(c).strip() for c in (data.get("concepts") or []) if str(c).strip()]
        relations = []
        concept_set = set(concepts)
        for r in data.get("relations") or []:
            if not isinstance(r, dict):
                continue
            s = str(r.get("source", "")).strip()
            t = str(r.get("target", "")).strip()
            rel = str(r.get("relation", "")).strip() or "связано с"
            if s and t and s in concept_set and t in concept_set:
                relations.append({"source": s, "relation": rel, "target": t})
        if concepts:
            return {"concepts": concepts[:16], "relations": relations[:20]}
    except Exception:
        pass
    return _fallback_concept_map(title, ru)


def merge_concept_maps(per_paper: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-paper concept maps into one graph payload (nodes + edges).

    Nodes are phrase concepts (deduplicated case-insensitively, weighted by how
    many papers mention them). Edges are named relations.
    """

    node_weight: Dict[str, float] = {}
    node_label: Dict[str, str] = {}
    edges: Dict[tuple, Dict[str, Any]] = {}

    for cm in per_paper:
        for c in cm.get("concepts", []) or []:
            key = c.strip().lower()
            if not key:
                continue
            node_weight[key] = node_weight.get(key, 0.0) + 1.0
            node_label.setdefault(key, c.strip())
        for r in cm.get("relations", []) or []:
            s = str(r.get("source", "")).strip().lower()
            t = str(r.get("target", "")).strip().lower()
            rel = str(r.get("relation", "")).strip() or "связано с"
            if not s or not t:
                continue
            node_weight.setdefault(s, node_weight.get(s, 1.0))
            node_weight.setdefault(t, node_weight.get(t, 1.0))
            node_label.setdefault(s, r.get("source", s))
            node_label.setdefault(t, r.get("target", t))
            ek = (s, t, rel)
            if ek in edges:
                edges[ek]["score"] += 1.0
            else:
                edges[ek] = {
                    "source": node_label[s],
                    "target": node_label[t],
                    "predicate": rel,
                    "score": 1.0,
                    "directed": True,
                }

    nodes = [
        {"term": node_label[k], "doc_freq": int(w)}
        for k, w in sorted(node_weight.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return {"nodes": nodes, "edges": list(edges.values()), "meta": {"kind": "concept_map"}}


def analyze_papers(
    papers: Sequence[Any],
    records_by_id: Dict[str, Any],
    *,
    use_llm: bool = True,
) -> List[Dict[str, Any]]:
    """Produce structured analysis + concept map for each paper.

    ``papers`` is a sequence of PaperMetadata; ``records_by_id`` maps paper id ->
    PaperRecord (for chunk-level RAG). Best-effort and offline-safe.
    """

    results: List[Dict[str, Any]] = []
    for p in papers:
        pid = getattr(p, "id", "")
        title = getattr(p, "title", "") or "(без названия)"
        abstract = (getattr(p, "abstract", "") or "").strip()
        # Переводим абстракт в русский ОДИН раз на статью: русская версия идёт в
        # офлайн-фолбэки структуры и графа понятий. Оригинал сохраняем для RAG и
        # дословного заземления цитат.
        abstract_ru = _to_russian(abstract) if abstract else abstract
        record = records_by_id.get(pid)

        if record is not None:
            passages = passages_from_record(record, abstract=abstract)
        else:
            passages = passages_from_record(
                type("R", (), {"evidence_units": [], "text": abstract})(), abstract=abstract
            )
        retriever = PaperRetriever(passages)

        structure = analyze_structure(
            title=title, abstract=abstract, abstract_ru=abstract_ru,
            retriever=retriever, use_llm=use_llm,
        )
        concept_map = extract_concept_map(
            title=title, abstract=abstract, abstract_ru=abstract_ru,
            retriever=retriever, use_llm=use_llm,
        )

        results.append(
            {
                "id": pid,
                "title": title,
                "structure": structure,
                "concept_map": concept_map,
                "n_passages": len(passages),
            }
        )
    return results
