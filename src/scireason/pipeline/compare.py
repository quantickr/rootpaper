# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Сравнение нескольких научных статей.

Пользователь присылает 2+ идентификатора (ссылку arXiv, DOI, URL или произвольный
запрос); мы загружаем метаданные каждой статьи (название + аннотация), а затем:

* строим текстовое сравнение (сходства / различия / связи) — LLM с детерминированным
  офлайн-fallback (в проде провайдер LLM = ``mock``, поэтому fallback обязателен);
* собираем таблицу сравнения (статьи — столбцы, аспекты — строки);
* строим граф связей между статьями (косинусная близость аннотаций, vis-network);
* выделяем общие темы/термины.

Результат — самодостаточный HTML (тёмная тема + vis-network с CDN), пригодный и для
показа на сайте, и для отправки файлом в Telegram.

Работаем **только по метаданным (title + abstract)**: аннотации почти всегда доступны,
PDF часто нет; так быстрее и надёжнее.
"""

import html
import json
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..papers.schema import PaperMetadata
from ..papers.service import get_paper_by_doi, search_papers

try:  # LLM — best-effort; сравнение должно собираться и офлайн.
    from ..llm import chat_json
except Exception:  # pragma: no cover - defensive import
    chat_json = None  # type: ignore[assignment]

# Переиспользуем готовые кирпичики отчёта: граф похожести и детектор mock-кода.
from .report import build_summary_graph, _looks_like_mock_code, _short_label


ProgressFn = Callable[[str, float], None]


# --------------------------------------------------------------------------- input parsing

_ARXIV_ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")
_ARXIV_OLD_RE = re.compile(r"\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})\b")
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE)


def _classify_token(token: str) -> Tuple[str, str]:
    """Определить тип идентификатора статьи.

    Возвращает пару ``(kind, value)`` где kind ∈ {doi, arxiv, search}.
    """

    t = (token or "").strip().strip("<>").rstrip(".,;)")
    if not t:
        return ("search", "")

    low = t.lower()

    # 1) DOI (в т.ч. в URL doi.org/…)
    m = _DOI_RE.search(t)
    if m and ("doi.org" in low or low.startswith("10.") or "doi:" in low):
        return ("doi", m.group(0).rstrip("."))

    # 2) arXiv (ссылка abs/pdf или голый id)
    if "arxiv.org" in low:
        m2 = _ARXIV_ID_RE.search(t) or _ARXIV_OLD_RE.search(t)
        if m2:
            return ("arxiv", m2.group(1))
    if low.startswith("arxiv:"):
        rest = t.split(":", 1)[1].strip()
        m2 = _ARXIV_ID_RE.search(rest) or _ARXIV_OLD_RE.search(rest)
        if m2:
            return ("arxiv", m2.group(1))

    # 3) Прочий URL — попробуем вытащить DOI из пути.
    if low.startswith("http://") or low.startswith("https://"):
        m3 = _DOI_RE.search(t)
        if m3:
            return ("doi", m3.group(0).rstrip("."))
        return ("search", t)

    # 4) Голый arXiv-id вида 2401.01234
    m4 = _ARXIV_ID_RE.fullmatch(t) or _ARXIV_OLD_RE.fullmatch(t)
    if m4:
        return ("arxiv", m4.group(1))

    # 5) Голый DOI без схемы
    if low.startswith("10.") and "/" in t:
        return ("doi", t)

    return ("search", t)


def parse_inputs(raw: str) -> List[Tuple[str, str]]:
    """Разобрать сырой ввод (много строк) в список ``(kind, value)``.

    Разделители — переводы строк; каждая непустая строка трактуется как один
    идентификатор статьи. Дубли убираем, сохраняя порядок.
    """

    tokens: List[str] = []
    for line in (raw or "").replace("\r", "\n").split("\n"):
        s = line.strip()
        if s:
            tokens.append(s)

    out: List[Tuple[str, str]] = []
    seen: set = set()
    for tok in tokens:
        kind, value = _classify_token(tok)
        if not value:
            continue
        key = (kind, value.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((kind, value))
    return out


# --------------------------------------------------------------------------- fetching

def fetch_paper(kind: str, value: str) -> Optional[PaperMetadata]:
    """Загрузить метаданные одной статьи по идентификатору. Best-effort."""

    try:
        if kind == "doi":
            return get_paper_by_doi(value)

        if kind == "arxiv":
            arxiv_id = value.strip()
            try:
                results = search_papers(arxiv_id, sources=["arxiv"], limit=5)
            except Exception:
                results = []
            # Ищем точное совпадение по arxiv-id, иначе берём первый результат.
            for p in results:
                pid = (getattr(p.ids, "arxiv", None) or "").lower()
                if pid and arxiv_id.lower() in pid:
                    return p
            if results:
                return results[0]
            # Фолбэк: обычный поиск без ограничения по источнику.
            try:
                results = search_papers(f"arXiv {arxiv_id}", limit=3)
            except Exception:
                results = []
            return results[0] if results else None

        # search: произвольный запрос/название → топ-1.
        try:
            results = search_papers(value, limit=1)
        except Exception:
            results = []
        return results[0] if results else None
    except Exception:
        return None


# --------------------------------------------------------------------------- common terms

# Короткие стоп-списки (EN + RU) для выделения общих терминов.
_STOPWORDS = {
    # EN
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "which", "have", "has", "not", "our", "their", "these", "those", "such",
    "using", "used", "use", "based", "study", "paper", "results", "result",
    "method", "methods", "model", "models", "approach", "propose", "proposed",
    "show", "shows", "shown", "also", "can", "may", "into", "than", "then",
    "more", "most", "each", "other", "between", "over", "under", "both", "via",
    "however", "while", "where", "when", "here", "they", "them", "its", "been",
    "abstract", "introduction", "conclusion", "data", "dataset", "datasets",
    # RU
    "это", "как", "для", "что", "или", "при", "также", "этот", "эта", "эти",
    "который", "которая", "которые", "быть", "было", "были", "может", "могут",
    "нами", "наша", "наши", "мы", "они", "его", "их", "все", "того", "этих",
    "статья", "статьи", "метод", "методы", "модель", "модели", "подход",
    "результат", "результаты", "исследование", "данные", "работа", "работе",
}

_TERM_RE = re.compile(r"[A-Za-z\u0400-\u04FF][A-Za-z0-9\u0400-\u04FF\-]{3,}")


def _paper_text(p: PaperMetadata) -> str:
    return f"{p.title or ''}. {p.abstract or ''}"


def common_terms(papers: List[PaperMetadata], *, top_k: int = 20) -> List[Tuple[str, int]]:
    """Термины, встречающиеся минимум в 2 статьях, отсортированные по частоте.

    Возвращает список ``(term, doc_freq)`` — сколько статей содержат термин.
    """

    per_paper: List[set] = []
    total_freq: Dict[str, int] = {}
    for p in papers:
        toks = {m.group(0).lower() for m in _TERM_RE.finditer(_paper_text(p))}
        toks = {t for t in toks if t not in _STOPWORDS}
        per_paper.append(toks)
        for t in toks:
            total_freq[t] = total_freq.get(t, 0) + 1

    # doc_freq: в скольких статьях встречается термин.
    doc_freq: Dict[str, int] = {}
    for toks in per_paper:
        for t in toks:
            doc_freq[t] = doc_freq.get(t, 0) + 1

    shared = [(t, df) for t, df in doc_freq.items() if df >= 2]
    shared.sort(key=lambda kv: (kv[1], total_freq.get(kv[0], 0)), reverse=True)
    return shared[:top_k]


# --------------------------------------------------------------------------- LLM comparison

_COMPARE_SCHEMA = (
    '{"similarities": ["строка", ...], "differences": ["строка", ...], '
    '"connections": ["строка", ...], "summary": "строка-вывод", '
    '"aspects": [{"name": "название аспекта", "values": ["значение для статьи 1", ...]}]}'
)


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip() + "\u2026"


def compare_with_llm(papers: List[PaperMetadata], terms: List[Tuple[str, int]]) -> Dict[str, Any]:
    """Сравнить статьи через LLM. При недоступности — детерминированный fallback."""

    if chat_json is not None:
        system = (
            "Ты — научный аналитик. Сравни несколько научных статей на РУССКОМ языке. "
            "Найди СХОДСТВА (общие идеи, методы, задачи), РАЗЛИЧИЯ (в подходах, данных, "
            "результатах) и СВЯЗИ (как статьи дополняют/развивают/противоречат друг другу). "
            "Сделай общий вывод. Также предложи 3–6 аспектов для таблицы сравнения: для "
            "каждого аспекта дай короткое значение по каждой статье (values в том же порядке, "
            "что и статьи). Пиши кратко и по делу, без markdown."
        )
        lines: List[str] = []
        for i, p in enumerate(papers, start=1):
            year = p.year or "—"
            abstract = _truncate(p.abstract or "", 1200)
            lines.append(f"[Статья {i}] {p.title} ({year})\nАннотация: {abstract or 'нет'}")
        user = "\n\n".join(lines)
        try:
            data = chat_json(system, user, _COMPARE_SCHEMA, temperature=0.2)
            norm = _normalize_llm_result(data, n_papers=len(papers))
            if norm is not None:
                return norm
        except Exception:
            pass

    return _fallback_comparison(papers, terms)


def _normalize_llm_result(data: Any, *, n_papers: int) -> Optional[Dict[str, Any]]:
    """Проверить и нормализовать ответ LLM. None — если ответ пустой/мок/мусор."""

    if not isinstance(data, dict) or not data:
        return None

    def _as_list(v: Any) -> List[str]:
        if isinstance(v, list):
            out = [str(x).strip() for x in v if str(x).strip()]
            return [s for s in out if not _looks_like_mock_code(s)]
        if isinstance(v, str) and v.strip() and not _looks_like_mock_code(v):
            return [v.strip()]
        return []

    similarities = _as_list(data.get("similarities"))
    differences = _as_list(data.get("differences"))
    connections = _as_list(data.get("connections"))
    summary = str(data.get("summary") or "").strip()
    if _looks_like_mock_code(summary):
        summary = ""

    aspects: List[Dict[str, Any]] = []
    raw_aspects = data.get("aspects")
    if isinstance(raw_aspects, list):
        for a in raw_aspects:
            if not isinstance(a, dict):
                continue
            name = str(a.get("name") or "").strip()
            values = a.get("values")
            if not name or not isinstance(values, list):
                continue
            vals = [str(x).strip() for x in values]
            # Выровнять длину под число статей.
            if len(vals) < n_papers:
                vals += ["—"] * (n_papers - len(vals))
            vals = vals[:n_papers]
            if any(_looks_like_mock_code(x) for x in vals):
                continue
            aspects.append({"name": name, "values": vals})

    # Если совсем пусто — считаем, что LLM ничего осмысленного не дал.
    if not (similarities or differences or connections or summary or aspects):
        return None

    return {
        "similarities": similarities,
        "differences": differences,
        "connections": connections,
        "summary": summary,
        "aspects": aspects,
        "source": "llm",
    }


def _fallback_comparison(
    papers: List[PaperMetadata], terms: List[Tuple[str, int]]
) -> Dict[str, Any]:
    """Детерминированное сравнение из метаданных (без LLM)."""

    n = len(papers)
    similarities: List[str] = []
    differences: List[str] = []
    connections: List[str] = []

    # Сходства — из общих терминов.
    top_terms = [t for t, _ in terms[:8]]
    if top_terms:
        similarities.append(
            "Общие темы и понятия во всех статьях: " + ", ".join(top_terms) + "."
        )
    else:
        similarities.append("Явных общих терминов в аннотациях не обнаружено.")

    # Различия — по годам, источникам, цитируемости.
    years = [p.year for p in papers if p.year]
    if years and (max(years) - min(years) >= 1):
        differences.append(
            f"Статьи опубликованы в разные годы: с {min(years)} по {max(years)}."
        )
    venues = {(_venue_name(p) or "").strip() for p in papers if _venue_name(p)}
    if len(venues) > 1:
        differences.append("Разные площадки/журналы публикации: " + ", ".join(sorted(venues)) + ".")
    cites = [(p.citation_count or 0) for p in papers]
    if cites and max(cites) > 0 and max(cites) != min(cites):
        differences.append(
            f"Заметная разница в цитируемости: от {min(cites)} до {max(cites)} ссылок."
        )
    if not differences:
        differences.append("Существенных формальных различий по метаданным не выявлено.")

    # Связи.
    if top_terms:
        connections.append(
            "Статьи связаны через общую тематику (" + ", ".join(top_terms[:5]) + "), "
            "поэтому их разумно изучать вместе."
        )
    if years:
        connections.append(
            "Более ранние работы могут служить основой, а более поздние — их развитием."
        )

    summary = (
        f"Сравнение {n} статей выполнено без LLM (офлайн-режим). Оно опирается на "
        "названия, аннотации и метаданные: общие термины, годы, источники и цитируемость."
    )

    # Аспекты-таблица из базовых полей.
    aspects = [
        {"name": "Год", "values": [str(p.year or "—") for p in papers]},
        {"name": "Источник", "values": [(_venue_name(p) or (p.source.value if p.source else "—")) for p in papers]},
        {"name": "Цитирований", "values": [str(p.citation_count if p.citation_count is not None else "—") for p in papers]},
        {"name": "Авторов", "values": [str(len(p.authors) if p.authors else 0) for p in papers]},
    ]

    return {
        "similarities": similarities,
        "differences": differences,
        "connections": connections,
        "summary": summary,
        "aspects": aspects,
        "source": "fallback",
    }


def _venue_name(p: PaperMetadata) -> Optional[str]:
    v = getattr(p, "venue", None)
    if v is None:
        return None
    name = getattr(v, "name", None)
    return str(name).strip() if name else None


# --------------------------------------------------------------------------- graph

def build_relation_graph(papers: List[PaperMetadata]) -> Dict[str, Any]:
    """vis-network payload {nodes, edges} со связями между статьями по близости."""

    records = [
        {
            "id": p.id,
            "title": p.title or "",
            "abstract": p.abstract or "",
            "summary": "",
            "citation_count": p.citation_count or 0,
        }
        for p in papers
        if p.id
    ]
    # Порог ниже дефолтного: статей мало, хотим видеть связи.
    graph = build_summary_graph(records, min_similarity=0.2, max_edges_per_node=4)

    vis_nodes = [
        {
            "id": nd["id"],
            "label": _short_label(nd.get("label", ""), 40),
            "value": float(nd.get("value", 1.0) or 1.0),
            "title": nd.get("title", ""),
        }
        for nd in graph.get("nodes", [])
    ]
    vis_edges = [
        {
            "from": e["source"],
            "to": e["target"],
            "value": max(0.5, float(e.get("score", 0.5) or 0.5)),
            "title": f"близость {float(e.get('score', 0.0)):.2f}",
        }
        for e in graph.get("edges", [])
    ]
    return {"nodes": vis_nodes, "edges": vis_edges}


# --------------------------------------------------------------------------- HTML

_COMPARE_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Сравнение статей</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  :root {{ --bg:#0f1116; --panel:#171a21; --border:#252a34; --text:#e6e6e6; --muted:#9aa0aa; --accent:#4fd1c5; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; background:var(--bg); color:var(--text); }}
  header {{ padding:18px 24px; border-bottom:1px solid var(--border); }}
  header h1 {{ margin:0 0 4px; font-size:20px; }}
  header .meta {{ color:var(--muted); font-size:13px; }}
  .wrap {{ padding:16px 24px; max-width:1200px; margin:0 auto; }}
  .card {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; margin:16px 0; overflow:hidden; }}
  .card h2 {{ margin:0; padding:12px 16px; font-size:15px; border-bottom:1px solid var(--border); }}
  .card .body {{ padding:12px 16px; }}
  ul.clean {{ margin:0; padding-left:20px; }}
  ul.clean li {{ margin:6px 0; line-height:1.5; font-size:14px; }}
  .cols3 {{ display:grid; grid-template-columns: 1fr 1fr 1fr; gap:16px; }}
  @media (max-width: 900px) {{ .cols3 {{ grid-template-columns: 1fr; }} }}
  .summary-box {{ font-size:14px; line-height:1.6; }}
  table.cmp {{ width:100%; border-collapse:collapse; font-size:13px; }}
  table.cmp th, table.cmp td {{ border:1px solid var(--border); padding:8px 10px; text-align:left; vertical-align:top; }}
  table.cmp th {{ background:#12151c; color:var(--text); }}
  table.cmp td.aspect {{ font-weight:600; background:#12151c; white-space:nowrap; }}
  .chips {{ display:flex; flex-wrap:wrap; gap:8px; }}
  .chip {{ background:linear-gradient(90deg,#3b82f6,#6366f1); color:#fff; border-radius:999px; padding:4px 12px; font-size:12px; font-weight:600; }}
  #graph {{ height:520px; background:#0b0d12; }}
  .paper {{ border:1px solid var(--border); border-radius:8px; margin:10px 0; padding:12px 14px; background:#12151c; }}
  .paper .ptitle {{ font-weight:600; font-size:14px; margin-bottom:6px; }}
  .paper .ptitle a {{ color:var(--accent); text-decoration:none; }}
  .paper .sub {{ color:var(--muted); font-size:12px; margin-top:4px; }}
  details {{ margin-top:8px; }}
  details > summary {{ cursor:pointer; color:var(--accent); font-size:13px; }}
  .abstract {{ color:#cfd3da; white-space:pre-wrap; font-size:13px; margin-top:6px; line-height:1.5; }}
  .badge {{ display:inline-block; font-size:11px; color:var(--muted); border:1px solid var(--border); border-radius:6px; padding:1px 6px; margin-left:6px; }}
  .warn {{ color:#f6ad55; font-size:13px; }}
</style>
</head>
<body>
<header>
  <h1>Сравнение статей</h1>
  <div class="meta">Статей: {n_papers} · {generated}{fallback_note}</div>
</header>
<div class="wrap">
  {failed_block}

  <div class="card"><h2>Вывод</h2><div class="body summary-box">{summary}</div></div>

  <div class="cols3">
    <div class="card"><h2>Сходства</h2><div class="body"><ul class="clean">{similarities}</ul></div></div>
    <div class="card"><h2>Различия</h2><div class="body"><ul class="clean">{differences}</ul></div></div>
    <div class="card"><h2>Связи</h2><div class="body"><ul class="clean">{connections}</ul></div></div>
  </div>

  <div class="card"><h2>Таблица сравнения</h2><div class="body" style="overflow-x:auto">{table}</div></div>

  <div class="card"><h2>Общие темы и термины</h2><div class="body"><div class="chips">{chips}</div></div></div>

  <div class="card"><h2>Граф связей</h2><div id="graph"></div></div>

  <div class="card"><h2>Статьи</h2><div class="body">{papers}</div></div>
</div>
<script>
const GRAPH = {graph_json};
(function() {{
  const container = document.getElementById('graph');
  if (!window.vis || !GRAPH || !GRAPH.nodes || !GRAPH.nodes.length) {{
    container.innerHTML = '<div style="padding:16px;color:#9aa0aa">Недостаточно данных для графа связей.</div>';
    return;
  }}
  const data = {{ nodes: new vis.DataSet(GRAPH.nodes), edges: new vis.DataSet(GRAPH.edges) }};
  const options = {{
    nodes: {{ shape: 'dot', scaling: {{ min: 8, max: 40 }}, font: {{ color: '#e6e6e6', size: 12 }}, color: {{ background: '#4fd1c5', border: '#2c7a7b', highlight: {{ background:'#f6e05e', border:'#b7791f' }} }} }},
    edges: {{ color: {{ color: '#3a4150', highlight: '#f6e05e' }}, smooth: {{ type: 'continuous' }}, scaling: {{ min: 0.5, max: 6 }} }},
    physics: {{ stabilization: {{ iterations: 150 }}, barnesHut: {{ gravitationalConstant: -8000, springLength: 130 }} }},
    interaction: {{ hover: true, tooltipDelay: 120 }},
  }};
  new vis.Network(container, data, options);
}})();
</script>
</body>
</html>
"""


def _esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _li_list(items: List[str]) -> str:
    if not items:
        return '<li class="warn">нет данных</li>'
    return "".join(f"<li>{_esc(x)}</li>" for x in items)


def _build_table(papers: List[PaperMetadata], aspects: List[Dict[str, Any]]) -> str:
    heads = "".join(
        f"<th>Статья {i}<div class='sub'>{_esc(_short_label(p.title or '', 36))}</div></th>"
        for i, p in enumerate(papers, start=1)
    )
    rows: List[str] = []
    for a in aspects:
        name = _esc(a.get("name", ""))
        vals = a.get("values", [])
        cells = "".join(f"<td>{_esc(v)}</td>" for v in vals)
        rows.append(f"<tr><td class='aspect'>{name}</td>{cells}</tr>")
    body = "".join(rows) or "<tr><td class='warn'>Нет аспектов для сравнения.</td></tr>"
    return f"<table class='cmp'><thead><tr><th>Аспект</th>{heads}</tr></thead><tbody>{body}</tbody></table>"


def _build_chips(terms: List[Tuple[str, int]]) -> str:
    if not terms:
        return '<span class="warn">Общих терминов не найдено.</span>'
    return "".join(f'<span class="chip">{_esc(t)} · {df}</span>' for t, df in terms)


def _build_paper_cards(papers: List[PaperMetadata]) -> str:
    cards: List[str] = []
    for i, p in enumerate(papers, start=1):
        title = _esc(p.title or "(без названия)")
        url = p.url or p.pdf_url
        title_html = f'<a href="{_esc(url)}" target="_blank" rel="noopener">{title}</a>' if url else title
        year = _esc(p.year or "—")
        venue = _esc(_venue_name(p) or (p.source.value if p.source else ""))
        cites = p.citation_count if p.citation_count is not None else "—"
        abstract = (p.abstract or "").strip()
        abstract_html = (
            f'<details><summary>Показать аннотацию</summary>'
            f'<div class="abstract">{_esc(abstract)}</div></details>'
            if abstract else '<div class="sub">Аннотация недоступна.</div>'
        )
        cards.append(
            f'<div class="paper"><div class="ptitle">{i}. {title_html}</div>'
            f'<div class="sub">Год: {year} · Источник: {venue} · Цитирований: {_esc(cites)}</div>'
            f'{abstract_html}</div>'
        )
    return "".join(cards)


def build_compare_html(
    papers: List[PaperMetadata],
    cmp: Dict[str, Any],
    terms: List[Tuple[str, int]],
    graph_payload: Dict[str, Any],
    *,
    failed: Optional[List[str]] = None,
) -> str:
    fallback_note = (
        " · офлайн-режим (без LLM)" if cmp.get("source") == "fallback" else ""
    )
    failed_block = ""
    if failed:
        items = "".join(f"<li>{_esc(x)}</li>" for x in failed)
        failed_block = (
            '<div class="card"><h2>⚠ Не удалось загрузить</h2>'
            f'<div class="body"><ul class="clean">{items}</ul></div></div>'
        )

    return _COMPARE_HTML_TEMPLATE.format(
        n_papers=len(papers),
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        fallback_note=fallback_note,
        failed_block=failed_block,
        summary=_esc(cmp.get("summary") or "Вывод недоступен."),
        similarities=_li_list(cmp.get("similarities") or []),
        differences=_li_list(cmp.get("differences") or []),
        connections=_li_list(cmp.get("connections") or []),
        table=_build_table(papers, cmp.get("aspects") or []),
        chips=_build_chips(terms),
        papers=_build_paper_cards(papers),
        graph_json=json.dumps(graph_payload, ensure_ascii=False),
    )


# --------------------------------------------------------------------------- orchestrator

class CompareError(Exception):
    """Понятная пользователю ошибка сравнения (мало статей и т.п.)."""


def run_compare(inputs: str, *, progress_fn: Optional[ProgressFn] = None) -> bytes:
    """Полный цикл сравнения: разбор → загрузка → LLM → термины/граф → HTML.

    Возвращает готовый самодостаточный HTML в байтах (UTF-8).
    Бросает :class:`CompareError`, если валидных статей меньше двух.
    """

    def _emit(label: str, frac: float) -> None:
        if progress_fn is not None:
            try:
                progress_fn(label, frac)
            except Exception:
                pass

    _emit("Разбор ссылок", 0.05)
    tokens = parse_inputs(inputs)
    if len(tokens) < 2:
        raise CompareError(
            "Нужно минимум 2 статьи. Пришли 2+ ссылки/DOI/arXiv-ID — по одной на строку."
        )

    papers: List[PaperMetadata] = []
    failed: List[str] = []
    total = len(tokens)
    for idx, (kind, value) in enumerate(tokens):
        frac = 0.10 + 0.45 * (idx / max(1, total))
        _emit(f"Загрузка статьи {idx + 1} из {total}", frac)
        p = fetch_paper(kind, value)
        if p is not None and (p.title or p.abstract):
            papers.append(p)
        else:
            failed.append(value)

    if len(papers) < 2:
        raise CompareError(
            "Удалось загрузить меньше двух статей. Проверь ссылки/DOI/arXiv-ID."
        )

    _emit("Выделение общих тем", 0.60)
    terms = common_terms(papers)

    _emit("Сравнение статей", 0.72)
    cmp = compare_with_llm(papers, terms)

    _emit("Построение графа связей", 0.88)
    graph_payload = build_relation_graph(papers)

    _emit("Формирование отчёта", 0.96)
    html_out = build_compare_html(papers, cmp, terms, graph_payload, failed=failed)

    _emit("Готово", 1.0)
    return html_out.encode("utf-8")
