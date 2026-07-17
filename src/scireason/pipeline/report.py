# SPDX-FileCopyrightText: 2026 top-papers-graph contributors
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

"""Per-run interactive HTML report.

Produces a single self-contained ``report.html`` for each pipeline run with:

* an interactive knowledge-graph map (vis-network, loaded from a CDN);
* a per-paper list with short summaries and collapsible "drill deeper" panels
  (abstract, authors, venue/year, source links, related graph terms).

The report is intentionally dependency-light: it emits plain HTML + a small
amount of vanilla JS and pulls the vis-network library from a CDN at view time.
No extra Python packages are required, so it always renders even when the
optional ``notebook_viz`` extras are not installed.
"""

import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..papers.schema import PaperMetadata

try:  # LLM is best-effort; the report must still render offline.
    from ..llm import chat_text
except Exception:  # pragma: no cover - defensive import
    chat_text = None  # type: ignore[assignment]

try:  # embeddings drive the paper-similarity ("summary") graph; hash fallback is offline-safe.
    from ..llm import embed
except Exception:  # pragma: no cover - defensive import
    embed = None  # type: ignore[assignment]

try:  # local offline EN->RU translator (+ critic); no-op if unavailable.
    from .translate import translate_ru
except Exception:  # pragma: no cover - defensive import
    translate_ru = None  # type: ignore[assignment]


def _to_russian(text: str) -> str:
    """Best-effort local translation to Russian; returns input unchanged on failure."""

    if translate_ru is None:
        return text
    try:
        return translate_ru(text)
    except Exception:
        return text


_WORD_RE = re.compile(r"[a-z0-9]+")

# The offline "mock" LLM provider returns deterministic tool-using PYTHON code
# (see scireason.llm.chat_text) instead of prose. If we ever get such a payload
# back for a summary, we must reject it and fall back to a real Russian summary.
_MOCK_MARKERS = (
    "# mock provider",
    "final_answer",
    "build_graph()",
    "graph_summary(",
    "cross_bridges(",
)


def _looks_like_mock_code(text: str) -> bool:
    """True if ``text`` is the mock provider's code payload, not a real summary."""

    t = (text or "").lower()
    return any(m in t for m in _MOCK_MARKERS)


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip()
    return f"{cut}\u2026"


_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _fallback_summary(paper: PaperMetadata) -> str:
    """Deterministic Russian-ish summary used when the LLM is unavailable.

    We keep a short, readable digest (first 1-2 sentences of the abstract) rather
    than dumping the whole abstract, so the offline card still reads like a
    summary. The abstract itself stays available under "Показать аннотацию".
    """

    abstract = (paper.abstract or "").strip()
    if abstract:
        sents = [s.strip() for s in _SENT_RE.split(abstract) if s.strip()]
        digest = " ".join(sents[:2]) if sents else abstract
        # Translate the English digest to Russian with a local model (if present).
        return _truncate(_to_russian(digest), 320)
    return _truncate(_to_russian(paper.title or "Аннотация недоступна."), 200)


def summarize_paper(paper: PaperMetadata, *, use_llm: bool = True) -> str:
    """Return a short (2-3 sentence) summary for a paper.

    Falls back to a truncated abstract if the LLM is unavailable or errors,
    so this never breaks an offline run.
    """

    if not use_llm or chat_text is None:
        return _fallback_summary(paper)

    abstract = (paper.abstract or "").strip()
    if not abstract:
        return _fallback_summary(paper)

    system = (
        "Ты — научный редактор. Кратко изложи статью в 2-3 предложениях на РУССКОМ языке "
        "для любознательного школьника. Скажи, что изучалось и какой ключевой результат. "
        "Без вступлений, без markdown, без списков."
    )
    user = f"Название: {paper.title}\n\nАннотация:\n{_truncate(abstract, 2000)}"
    try:
        out = chat_text(system, user, temperature=0.2)
        out = (out or "").strip()
        # Reject the offline mock provider's code payload (it is not a summary).
        if out and not _looks_like_mock_code(out):
            return _truncate(out, 600)
    except Exception:
        pass
    return _fallback_summary(paper)


def summarize_papers(
    papers: Sequence[PaperMetadata], *, use_llm: bool = True
) -> List[Dict[str, Any]]:
    """Build the per-paper summary records used by the HTML dashboard."""

    records: List[Dict[str, Any]] = []
    for p in papers:
        authors = [a.name for a in (p.authors or []) if getattr(a, "name", None)]
        venue = getattr(p.venue, "name", None) if p.venue else None
        records.append(
            {
                "id": p.id,
                "title": p.title or "(untitled)",
                "summary": summarize_paper(p, use_llm=use_llm),
                "abstract": (p.abstract or "").strip(),
                "authors": authors,
                "venue": venue,
                "year": p.year,
                "citation_count": p.citation_count,
                "url": p.url,
                "pdf_url": p.pdf_url,
                "doi": getattr(p.ids, "doi", None) if p.ids else None,
                "source": str(p.source),
            }
        )
    return records


def _paper_terms(records: Sequence[Dict[str, Any]], graph: Dict[str, Any]) -> None:
    """Annotate each paper record with related graph terms (in place).

    Matches graph node terms that appear in the paper's title/abstract. This is
    a lightweight lexical match used purely for the "drill deeper" UI.
    """

    node_terms = []
    for n in graph.get("nodes", []) or []:
        if isinstance(n, dict):
            term = n.get("term") or n.get("id") or n.get("label")
            if term:
                node_terms.append(str(term))

    for rec in records:
        haystack = f"{rec.get('title', '')} {rec.get('abstract', '')}".lower()
        hay_tokens = set(_WORD_RE.findall(haystack))
        related: List[str] = []
        for term in node_terms:
            tl = term.lower().strip()
            if not tl:
                continue
            if " " in tl:
                if tl in haystack:
                    related.append(term)
            elif tl in hay_tokens:
                related.append(term)
            if len(related) >= 8:
                break
        rec["related_terms"] = related


def _graph_view_payload(
    graph: Dict[str, Any], *, max_nodes: int = 200, show_edge_labels: bool = True
) -> Dict[str, Any]:
    """Reduce the temporal KG JSON into a compact vis-network payload.

    Keeps the highest-degree nodes (bounded by ``max_nodes``) to keep the
    browser view responsive on large graphs. ``show_edge_labels=False`` hides the
    on-canvas edge caption (used for the paper-similarity graph, where a repeated
    "похожа по смыслу" label would clutter the view); the relation still appears
    in the edge tooltip.
    """

    raw_edges = [e for e in (graph.get("edges") or []) if isinstance(e, dict)]

    # Degree by edge endpoints (edges are the source of truth for connectivity).
    degree: Dict[str, float] = {}
    for e in raw_edges:
        s, t = e.get("source"), e.get("target")
        w = float(e.get("score", e.get("total_count", 1.0)) or 1.0)
        if s:
            degree[str(s)] = degree.get(str(s), 0.0) + w
        if t:
            degree[str(t)] = degree.get(str(t), 0.0) + w

    # Node identity: prefer an explicit ``id`` (paper-similarity graph), else the
    # ``term`` (temporal KG). We keep the display ``label`` separately so paper
    # nodes can show a readable title while edges still reference the paper id.
    node_terms: Dict[str, Dict[str, Any]] = {}
    for n in graph.get("nodes", []) or []:
        if isinstance(n, dict):
            key = n.get("id") or n.get("term") or n.get("label")
            if key:
                node_terms[str(key)] = n

    ordered = sorted(degree.items(), key=lambda kv: kv[1], reverse=True)
    kept = {term for term, _ in ordered[:max_nodes]}
    # Include isolated nodes only if there is room left.
    if len(kept) < max_nodes:
        for term in node_terms:
            if term not in kept:
                kept.add(term)
            if len(kept) >= max_nodes:
                break

    vis_nodes: List[Dict[str, Any]] = []
    for term in kept:
        stats = node_terms.get(term, {})
        deg = degree.get(term, 0.0)
        label = str(stats.get("label") or stats.get("term") or term)
        base_value = float(stats.get("value", 0.0) or 0.0)
        tooltip = str(stats.get("title") or f"{label} (weight={deg:.2f}, doc_freq={stats.get('doc_freq', 0)})")
        vis_nodes.append(
            {
                "id": term,
                "label": label,
                "value": max(1.0, deg, base_value),
                "title": tooltip,
            }
        )

    vis_edges: List[Dict[str, Any]] = []
    for e in raw_edges:
        s, t = e.get("source"), e.get("target")
        if not s or not t:
            continue
        s, t = str(s), str(t)
        if s not in kept or t not in kept:
            continue
        w = float(e.get("score", e.get("total_count", 1.0)) or 1.0)
        pred = str(e.get("predicate", "") or "")
        s_label = str(node_terms.get(s, {}).get("label") or s)
        t_label = str(node_terms.get(t, {}).get("label") or t)
        vis_edges.append(
            {
                "from": s,
                "to": t,
                "value": max(0.5, w),
                "label": pred if show_edge_labels else "",
                "title": f"{s_label} — {pred} — {t_label} (score={w:.2f})",
                "arrows": "to" if e.get("directed", True) else "",
            }
        )

    return {"nodes": vis_nodes, "edges": vis_edges}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = na = nb = 0.0
    for x, y in zip(a, b):
        num += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return num / ((na ** 0.5) * (nb ** 0.5))


def _short_label(title: str, limit: int = 42) -> str:
    title = (title or "").strip() or "(без названия)"
    return title if len(title) <= limit else title[:limit].rsplit(" ", 1)[0].rstrip() + "\u2026"


def build_summary_graph(
    records: Sequence[Dict[str, Any]],
    *,
    min_similarity: float = 0.35,
    max_edges_per_node: int = 4,
) -> Dict[str, Any]:
    """Build a paper-similarity graph from the *summaries* (not the term KG).

    Nodes are papers; edges connect papers whose summary/abstract are
    semantically close (cosine over embeddings, with a deterministic hash-embed
    fallback so it works fully offline). This is the second, "суммаризация" map.
    """

    items = [r for r in records if r.get("id")]
    if len(items) < 2:
        return {"nodes": [], "edges": [], "meta": {"kind": "summary_graph"}}

    texts = [
        f"{r.get('title', '')}. {r.get('summary', '')} {r.get('abstract', '')}".strip()
        for r in items
    ]

    vectors: List[List[float]] = []
    if embed is not None:
        try:
            vectors = embed(texts)
        except Exception:
            vectors = []

    # Build similarity edges first: we need node connectivity to compute the
    # recommended study order ("от обзорных к узким").
    edges: List[Dict[str, Any]] = []
    if vectors and len(vectors) == len(items):
        n = len(items)
        # For each paper keep its top-K most similar neighbours above threshold.
        for i in range(n):
            sims = []
            for j in range(n):
                if i == j:
                    continue
                s = _cosine(vectors[i], vectors[j])
                if s >= min_similarity:
                    sims.append((j, s))
            sims.sort(key=lambda kv: kv[1], reverse=True)
            for j, s in sims[:max_edges_per_node]:
                a, b = str(items[i]["id"]), str(items[j]["id"])
                key = (a, b) if a < b else (b, a)
                edges.append(
                    {
                        "source": key[0],
                        "target": key[1],
                        "predicate": "похожа по смыслу",
                        "score": round(float(s), 3),
                        "directed": False,
                    }
                )

    # Deduplicate undirected edges (keep the highest score).
    dedup: Dict[tuple, Dict[str, Any]] = {}
    for e in edges:
        k = (e["source"], e["target"])
        if k not in dedup or e["score"] > dedup[k]["score"]:
            dedup[k] = e
    final_edges = list(dedup.values())

    # --- Recommended study order: broad/overview papers first, then narrower. ---
    # A paper is treated as more "overview" the more it is connected to the rest
    # (weighted degree in the similarity graph) and the more it is cited. We rank
    # by that combined score and number the nodes 1..N so students see where to
    # start and how to proceed.
    connectivity: Dict[str, float] = {str(r["id"]): 0.0 for r in items}
    for e in final_edges:
        connectivity[e["source"]] = connectivity.get(e["source"], 0.0) + e["score"]
        connectivity[e["target"]] = connectivity.get(e["target"], 0.0) + e["score"]

    max_cites = max((float(r.get("citation_count") or 0) for r in items), default=0.0)

    def _study_score(r: Dict[str, Any]) -> float:
        pid = str(r["id"])
        deg = connectivity.get(pid, 0.0)
        cites = float(r.get("citation_count") or 0)
        # Normalise citations to the graph so it complements (not dominates) degree.
        cite_norm = (cites / max_cites) if max_cites > 0 else 0.0
        return deg + 0.5 * cite_norm

    ordered_items = sorted(
        items,
        key=lambda r: (_study_score(r), float(r.get("citation_count") or 0)),
        reverse=True,
    )
    order_by_id: Dict[str, int] = {
        str(r["id"]): idx for idx, r in enumerate(ordered_items, start=1)
    }

    nodes: List[Dict[str, Any]] = []
    for r in items:
        pid = str(r["id"])
        order = order_by_id.get(pid, 0)
        base_title = r.get("title", "")
        nodes.append(
            {
                "id": pid,
                "label": f"{order}. {_short_label(base_title)}",
                "value": float(r.get("citation_count") or 0) or 1.0,
                "title": f"#{order} для изучения — {base_title}",
                "order": order,
            }
        )

    # Keep nodes sorted by study order so downstream JSON/consumers see 1..N.
    nodes.sort(key=lambda nd: nd.get("order", 0))

    return {
        "nodes": nodes,
        "edges": final_edges,
        "meta": {"kind": "summary_graph", "ordered": "study"},
    }


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  :root {{ --bg:#0f1116; --panel:#171a21; --border:#252a34; --text:#e6e6e6; --muted:#9aa0aa; --accent:#4fd1c5; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; background:var(--bg); color:var(--text); }}
  header {{ padding:18px 24px; border-bottom:1px solid var(--border); }}
  header h1 {{ margin:0 0 4px; font-size:20px; }}
  header .meta {{ color:var(--muted); font-size:13px; }}
  .layout {{ display:grid; grid-template-columns: 1fr 1fr; gap:16px; padding:16px 24px; }}
  @media (max-width: 1000px) {{ .layout {{ grid-template-columns: 1fr; }} }}
  .card {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
  .card h2 {{ margin:0; padding:12px 16px; font-size:15px; border-bottom:1px solid var(--border); }}
  #graph {{ height:620px; background:#0b0d12; }}
  .papers {{ max-height:620px; overflow-y:auto; padding:8px 12px; }}
  .paper {{ border:1px solid var(--border); border-radius:8px; margin:10px 0; padding:12px 14px; background:#12151c; }}
  .paper .ptitle {{ font-weight:600; font-size:14px; margin-bottom:6px; }}
  .paper .summary {{ color:var(--text); font-size:13px; line-height:1.5; }}
  .paper .sub {{ color:var(--muted); font-size:12px; margin-top:6px; }}
  details {{ margin-top:8px; }}
  details > summary {{ cursor:pointer; color:var(--accent); font-size:13px; user-select:none; }}
  .detail-body {{ margin-top:8px; font-size:13px; line-height:1.5; }}
  .detail-body .abstract {{ color:#cfd3da; white-space:pre-wrap; }}
  .kv {{ color:var(--muted); font-size:12px; margin-top:6px; }}
  .tags span {{ display:inline-block; background:#1d2a2a; color:var(--accent); border-radius:12px; padding:2px 8px; margin:2px 4px 2px 0; font-size:11px; }}
  a {{ color:var(--accent); }}
  .links a {{ margin-right:12px; font-size:12px; }}
  .search {{ padding:8px 12px; }}
  .search input {{ width:100%; padding:8px 10px; background:#0b0d12; border:1px solid var(--border); border-radius:6px; color:var(--text); }}
  .toggle {{ display:flex; gap:6px; }}
  .toggle button {{ background:#0b0d12; color:var(--muted); border:1px solid var(--border); border-radius:6px; padding:4px 10px; font-size:12px; cursor:pointer; }}
  .toggle button.active {{ color:#0b0d12; background:var(--accent); border-color:var(--accent); }}
  .cardhead {{ display:flex; align-items:center; justify-content:space-between; padding:12px 16px; border-bottom:1px solid var(--border); }}
  .cardhead h2 {{ margin:0; padding:0; border:0; font-size:15px; }}
  .struct {{ margin-top:10px; }}
  .struct .row {{ margin:6px 0; }}
  .struct .flabel {{ color:var(--accent); font-weight:600; font-size:12px; }}
  .struct .ftext {{ font-size:13px; line-height:1.45; }}
  .badge {{ display:inline-block; font-size:10px; border-radius:8px; padding:1px 6px; margin-left:6px; vertical-align:middle; }}
  .badge.ok {{ background:#14331f; color:#68d391; }}
  .badge.warn {{ background:#3a2a14; color:#f6ad55; }}
  .quote {{ color:#9aa0aa; font-size:12px; border-left:2px solid var(--border); padding-left:8px; margin-top:2px; font-style:italic; }}
  .plan {{ margin:6px 0 0 18px; font-size:13px; line-height:1.5; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="meta">{subtitle}</div>
</header>
<div class="layout">
  <div class="card">
    <div class="cardhead">
      <h2>Карта графа</h2>
      <div class="toggle" id="graphToggle">
        <button data-mode="kg" class="active">Понятия (смысловые выражения)</button>
        <button data-mode="summary">Суммаризация (порядок изучения 1→N)</button>
      </div>
    </div>
    <div id="graph"></div>
  </div>
  <div class="card">
    <h2 style="padding:12px 16px;border-bottom:1px solid var(--border)">Статьи &middot; {n_papers}</h2>
    <div class="search"><input id="paperSearch" placeholder="Фильтр по названию или саммари\u2026"/></div>
    <div class="papers" id="papers"></div>
  </div>
</div>
<script>
const GRAPH = {graph_json};
const SUMMARY_GRAPH = {summary_json};
const PAPERS = {papers_json};

// ---- Interactive graph map with KG / concept-map toggle ----
(function() {{
  const container = document.getElementById('graph');
  let network = null;
  function draw(payload) {{
    if (!window.vis || !payload || !payload.nodes.length) {{
      container.innerHTML = '<div style="padding:16px;color:#9aa0aa">Нет данных для отображения.</div>';
      return;
    }}
    const data = {{ nodes: new vis.DataSet(payload.nodes), edges: new vis.DataSet(payload.edges) }};
    const options = {{
      nodes: {{ shape: 'dot', scaling: {{ min: 6, max: 40 }}, font: {{ color: '#e6e6e6', size: 12 }}, color: {{ background: '#4fd1c5', border: '#2c7a7b', highlight: {{ background:'#f6e05e', border:'#b7791f' }} }} }},
      edges: {{ color: {{ color: '#3a4150', highlight: '#f6e05e' }}, smooth: {{ type: 'continuous' }}, font: {{ color: '#8a909a', size: 10, strokeWidth: 0 }}, scaling: {{ min: 0.5, max: 6 }} }},
      physics: {{ stabilization: {{ iterations: 150 }}, barnesHut: {{ gravitationalConstant: -8000, springLength: 120 }} }},
      interaction: {{ hover: true, tooltipDelay: 120 }},
    }};
    if (network) {{ network.destroy(); }}
    network = new vis.Network(container, data, options);
  }}
  draw(GRAPH);
  document.querySelectorAll('#graphToggle button').forEach(btn => {{
    btn.addEventListener('click', () => {{
      document.querySelectorAll('#graphToggle button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      draw(btn.dataset.mode === 'summary' ? SUMMARY_GRAPH : GRAPH);
    }});
  }});
}})();

// ---- Paper list with structured analysis + drill-down ----
(function() {{
  const host = document.getElementById('papers');
  function esc(s) {{ const d = document.createElement('div'); d.textContent = (s == null ? '' : String(s)); return d.innerHTML; }}
  const LABELS = {{ hypothesis:'Гипотеза', method:'Метод', data:'Данные', results:'Результаты', limitations:'Ограничения' }};
  function structHtml(st) {{
    if (!st) return '';
    let h = '<div class="struct">';
    Object.keys(LABELS).forEach(k => {{
      const f = st[k]; if (!f) return;
      const badge = f.quote ? (f.verified
        ? '<span class="badge ok" title="Цитата найдена в тексте">✓ источник' + (f.page!=null?(' стр.'+f.page):'') + '</span>'
        : '<span class="badge warn" title="Цитата не подтверждена в тексте">⚠ не подтверждено</span>') : '';
      h += '<div class="row"><span class="flabel">' + esc(LABELS[k]) + ':</span> <span class="ftext">' + esc(f.text || 'н/д') + '</span>' + badge +
           (f.quote ? '<div class="quote">«' + esc(f.quote) + '»</div>' : '') + '</div>';
    }});
    h += '</div>';
    return h;
  }}
  function render(list) {{
    host.innerHTML = '';
    list.forEach(p => {{
      const el = document.createElement('div');
      el.className = 'paper';
      const authors = (p.authors || []).slice(0, 6).join(', ') + ((p.authors || []).length > 6 ? ', и др.' : '');
      const sub = [authors, p.venue, p.year, (p.citation_count != null ? p.citation_count + ' цит.' : null)].filter(Boolean).map(esc).join(' &middot; ');
      const tags = (p.related_terms || []).map(t => '<span>' + esc(t) + '</span>').join('');
      const links = [];
      if (p.url) links.push('<a href="' + esc(p.url) + '" target="_blank" rel="noopener">Источник</a>');
      if (p.pdf_url) links.push('<a href="' + esc(p.pdf_url) + '" target="_blank" rel="noopener">PDF</a>');
      if (p.doi) links.push('<a href="https://doi.org/' + esc(p.doi) + '" target="_blank" rel="noopener">DOI</a>');
      el.innerHTML =
        '<div class="ptitle">' + esc(p.title) + '</div>' +
        '<div class="summary">' + esc(p.summary) + '</div>' +
        (sub ? '<div class="sub">' + sub + '</div>' : '') +
        '<details><summary>Разобрать статью</summary><div class="detail-body">' +
          structHtml(p.structure) +
          (p.abstract ? '<details style="margin-top:8px"><summary>Показать аннотацию</summary><div class="abstract" style="margin-top:6px">' + esc(p.abstract) + '</div></details>' : '') +
          (tags ? '<div class="tags" style="margin-top:8px">' + tags + '</div>' : '') +
          (links.length ? '<div class="links" style="margin-top:8px">' + links.join('') + '</div>' : '') +
        '</div></details>';
      host.appendChild(el);
    }});
  }}
  render(PAPERS);
  const search = document.getElementById('paperSearch');
  search.addEventListener('input', () => {{
    const q = search.value.trim().toLowerCase();
    if (!q) {{ render(PAPERS); return; }}
    render(PAPERS.filter(p => ((p.title || '') + ' ' + (p.summary || '')).toLowerCase().includes(q)));
  }});
}})();
</script>
</body>
</html>
"""


def build_report_html(
    *,
    query: str,
    domain_title: str,
    graph: Dict[str, Any],
    paper_records: List[Dict[str, Any]],
    concept_graph: Optional[Dict[str, Any]] = None,
    max_graph_nodes: int = 200,
) -> str:
    """Assemble the self-contained report HTML string.

    ``paper_records`` may already carry a ``related_terms`` key; if not, it is
    computed here so the HTML is self-consistent.
    """

    if paper_records and "related_terms" not in paper_records[0]:
        _paper_terms(paper_records, graph)

    # First map: concept map of meaningful *phrases* (смысловые выражения) with
    # named relations. Fall back to the temporal term graph only if the concept
    # map is empty (e.g. analysis unavailable).
    first_graph = concept_graph if (concept_graph and concept_graph.get("nodes")) else graph
    graph_payload = _graph_view_payload(first_graph, max_nodes=max_graph_nodes)

    # Second map: paper-similarity graph built from the summaries themselves.
    # Hide the repeated "похожа по смыслу" edge caption to keep the view clean.
    summary_graph = build_summary_graph(paper_records)
    summary_payload = _graph_view_payload(
        summary_graph, max_nodes=max_graph_nodes, show_edge_labels=False
    )

    title = html.escape(f"Разбор статей: {query}")
    subtitle = html.escape(
        f"Домен: {domain_title} | {len(graph_payload['nodes'])} понятий, "
        f"{len(graph_payload['edges'])} связей | "
        f"граф суммаризации: {len(summary_payload['nodes'])} статей, "
        f"{len(summary_payload['edges'])} связей"
    )

    return _HTML_TEMPLATE.format(
        title=title,
        subtitle=subtitle,
        n_papers=len(paper_records),
        graph_json=json.dumps(graph_payload, ensure_ascii=False),
        summary_json=json.dumps(summary_payload, ensure_ascii=False),
        papers_json=json.dumps(paper_records, ensure_ascii=False),
    )


def write_run_report(
    *,
    out_dir: Path,
    query: str,
    domain_title: str,
    papers: Sequence[PaperMetadata],
    graph_json_path: Path,
    records_by_id: Optional[Dict[str, Any]] = None,
    use_llm: bool = True,
    max_graph_nodes: int = 200,
) -> Dict[str, Path]:
    """Generate the interactive run report and its data artifacts.

    Produces: ``report.html`` (graph map + concept map + per-paper structured
    analysis), ``paper_summaries.json``, ``paper_analysis.json`` (structured
    breakdown + citations) and ``concept_map.json`` (merged concept graph).

    Returns a mapping of artifact name -> path. Best-effort: never raises on
    summary/analysis issues; missing graph data yields an empty graph view.
    """

    try:
        graph = json.loads(Path(graph_json_path).read_text(encoding="utf-8"))
    except Exception:
        graph = {"nodes": [], "edges": []}

    records = summarize_papers(papers, use_llm=use_llm)
    # Annotate related graph terms before persisting so the JSON artifact and the
    # HTML dashboard show the same "drill deeper" data.
    _paper_terms(records, graph)

    # Structured, RAG-grounded analysis + phrase-level concept map (best-effort).
    concept_graph: Dict[str, Any] = {"nodes": [], "edges": []}
    analyses: List[Dict[str, Any]] = []
    try:
        from .paper_analysis import analyze_papers, merge_concept_maps

        analyses = analyze_papers(papers, records_by_id or {}, use_llm=use_llm)
        concept_graph = merge_concept_maps([a.get("concept_map", {}) for a in analyses])

        # Attach the structured breakdown to each paper record (by id) for the UI.
        struct_by_id = {a["id"]: a.get("structure", {}) for a in analyses}
        for rec in records:
            rec["structure"] = struct_by_id.get(rec.get("id"), {})

        (out_dir / "paper_analysis.json").write_text(
            json.dumps(analyses, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "concept_map.json").write_text(
            json.dumps(concept_graph, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        # Analysis is optional; the report still renders with summaries only.
        pass

    summaries_path = out_dir / "paper_summaries.json"
    summaries_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Persist the paper-similarity ("summarization") graph as a data artifact.
    summary_graph = build_summary_graph(records)
    (out_dir / "summary_graph.json").write_text(
        json.dumps(summary_graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_path = out_dir / "report.html"
    report_path.write_text(
        build_report_html(
            query=query,
            domain_title=domain_title,
            graph=graph,
            paper_records=records,
            concept_graph=concept_graph,
            max_graph_nodes=max_graph_nodes,
        ),
        encoding="utf-8",
    )

    artifacts: Dict[str, Path] = {
        "report": report_path,
        "summaries": summaries_path,
        "summary_graph": out_dir / "summary_graph.json",
    }
    if analyses:
        artifacts["analysis"] = out_dir / "paper_analysis.json"
        artifacts["concept_map"] = out_dir / "concept_map.json"
    return artifacts
