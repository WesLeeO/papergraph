"""The agent's tools: what it can do in the world, described for the model.

Each tool is a JSON-schema description the model reads (name, purpose,
arguments) plus the Python function that runs it. Results go back to the model
as compact text (every token counts in a small model's context) and to the UI
as structured paper records.
"""

import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import psycopg

from papergraph import search as svc
from papergraph.retrieval.filters import Filters
from papergraph.search import Paper

ABSTRACT_CHARS = 350  # per paper in tool results; get_paper returns the full abstract


@dataclass
class ToolResult:
    text: str  # what the model reads
    papers: list[Paper] = field(default_factory=list)  # what the UI shows


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema
    run: Callable[..., ToolResult]

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def format_paper(p: Paper, full: bool = False) -> str:
    year = (p.published or "")[:4]
    venue = f", {p.venue} {p.venue_year or ''}".rstrip() if p.venue else ""
    authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
    abstract = p.abstract if full else p.abstract[:ABSTRACT_CHARS]
    if not full and len(p.abstract) > ABSTRACT_CHARS:
        abstract += "..."
    return f"[{p.id}] {p.title} ({year}{venue}) — {authors}\n    {abstract}"


def format_papers(papers: list[Paper]) -> str:
    return "\n".join(format_paper(p) for p in papers) if papers else "No papers found."


def clamp(value: Any, low: int, high: int, default: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def as_list(value: Any) -> tuple[str, ...]:
    """Models send lists, comma-separated strings or single strings: accept all."""
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        value = value.split(",")
    return tuple(str(v).strip() for v in value if str(v).strip())


# --- arXiv live search ----------------------------------------------------------

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_MIN_INTERVAL_S = 3.0  # arXiv asks API clients for at most one request every 3 s
_ATOM = {"a": "http://www.w3.org/2005/Atom"}
_last_arxiv_call = 0.0


def arxiv_query(text: str) -> str:
    """'sparse attention for long docs' -> 'all:sparse AND all:attention AND ...'."""
    words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", text) if len(w) > 2][:6]
    return " AND ".join(f"all:{w}" for w in words)


def parse_arxiv_feed(xml_text: str) -> list[Paper]:
    papers = []
    for entry in ET.fromstring(xml_text).findall("a:entry", _ATOM):
        raw_id = (entry.findtext("a:id", "", _ATOM) or "").rsplit("/abs/", 1)[-1]
        paper_id = re.sub(r"v\d+$", "", raw_id)  # 2511.01234v2 -> 2511.01234
        papers.append(
            Paper(
                id=paper_id,
                title=" ".join((entry.findtext("a:title", "", _ATOM) or "").split()),
                abstract=" ".join((entry.findtext("a:summary", "", _ATOM) or "").split()),
                authors=[
                    a.findtext("a:name", "", _ATOM) or "" for a in entry.findall("a:author", _ATOM)
                ],
                published=(entry.findtext("a:published", "", _ATOM) or "")[:10] or None,
                venue=None,
                venue_year=None,
                categories=[c.get("term", "") for c in entry.findall("a:category", _ATOM)],
            )
        )
    return papers


def fetch_arxiv(query: str, max_results: int, http: httpx.Client) -> list[Paper]:
    global _last_arxiv_call
    wait = ARXIV_MIN_INTERVAL_S - (time.monotonic() - _last_arxiv_call)
    if wait > 0:
        time.sleep(wait)
    _last_arxiv_call = time.monotonic()
    resp = http.get(
        ARXIV_API,
        params={
            "search_query": arxiv_query(query),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max_results,
        },
    )
    resp.raise_for_status()
    return parse_arxiv_feed(resp.text)


# --- the toolbox ------------------------------------------------------------------


class Toolbox:
    """The agent's tools, bound to a database connection."""

    def __init__(self, conn: psycopg.Connection, http: httpx.Client | None = None) -> None:
        self.conn = conn
        self.http = http or httpx.Client(
            timeout=20, headers={"User-Agent": "PaperGraph research agent (educational project)"}
        )
        self.tools = {t.name: t for t in self._build()}

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self.tools.values()]

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run a tool; errors come back as text the model can read and correct."""
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(f"Error: unknown tool {name!r}. Available: {', '.join(self.tools)}.")
        required = tool.parameters.get("required", [])
        missing = [a for a in required if not arguments.get(a)]
        if missing:
            return ToolResult(f"Error: {name} needs argument(s) {', '.join(missing)}.")
        allowed = set(tool.parameters.get("properties", {}))
        args = {k: v for k, v in arguments.items() if k in allowed}  # drop invented arguments
        try:
            return tool.run(**args)
        except Exception as e:  # report to the model instead of crashing the loop
            return ToolResult(f"Error running {name}: {type(e).__name__}: {e}")

    # Tool implementations -------------------------------------------------------

    def search_papers(
        self,
        query: str,
        year_from: Any = None,
        year_to: Any = None,
        venues: Any = None,
        k: Any = 8,
    ) -> ToolResult:
        filters = Filters(
            year_from=clamp(year_from, 1990, 2100, 0) or None,
            year_to=clamp(year_to, 1990, 2100, 0) or None,
            venues=as_list(venues),
        )
        papers = svc.search(self.conn, str(query), k=clamp(k, 1, 15, 8), filters=filters)
        return ToolResult(format_papers(papers), papers)

    def get_paper(self, paper_id: str) -> ToolResult:
        paper = svc.get_paper(self.conn, str(paper_id).strip("[] "))
        if paper is None:
            return ToolResult(f"No paper with id {paper_id!r} in the local index.")
        return ToolResult(format_paper(paper, full=True), [paper])

    def similar_papers(self, paper_id: str, k: Any = 8) -> ToolResult:
        papers = svc.similar_papers(self.conn, str(paper_id).strip("[] "), k=clamp(k, 1, 15, 8))
        if not papers:
            return ToolResult(f"No embedding for paper {paper_id!r}; try search_papers instead.")
        return ToolResult(format_papers(papers), papers)

    def search_arxiv(self, query: str, max_results: Any = 5) -> ToolResult:
        papers = fetch_arxiv(str(query), clamp(max_results, 1, 10, 5), self.http)
        return ToolResult(format_papers(papers), papers)

    def _build(self) -> list[Tool]:
        return [
            Tool(
                "search_papers",
                "Search the local index of ~581k arXiv ML papers (up to late 2026). "
                "Main tool. Use short, specific queries.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "short search query"},
                        "year_from": {"type": "integer", "description": "earliest year"},
                        "year_to": {"type": "integer", "description": "latest year"},
                        "venues": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "only these venues, e.g. ['ICLR', 'NeurIPS']",
                        },
                        "k": {"type": "integer", "description": "number of results (max 15)"},
                    },
                    "required": ["query"],
                },
                self.search_papers,
            ),
            Tool(
                "get_paper",
                "Get one paper's full abstract and metadata by its id.",
                {
                    "type": "object",
                    "properties": {"paper_id": {"type": "string"}},
                    "required": ["paper_id"],
                },
                self.get_paper,
            ),
            Tool(
                "similar_papers",
                "Find papers most similar to a given paper (by embedding).",
                {
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string"},
                        "k": {"type": "integer", "description": "number of results (max 15)"},
                    },
                    "required": ["paper_id"],
                },
                self.similar_papers,
            ),
            Tool(
                "search_arxiv",
                "Live search of arXiv for the newest preprints, newest first. "
                "Slow; only for very recent work.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "a few key terms"},
                        "max_results": {"type": "integer", "description": "max 10"},
                    },
                    "required": ["query"],
                },
                self.search_arxiv,
            ),
        ]
