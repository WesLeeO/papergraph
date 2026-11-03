"""Metadata filters (year, venue, category) applied INSIDE each retriever.

Filtering after retrieval would break: ask the dense index for 100 neighbours,
then keep "ICLR only", and you may be left with 3 papers. So each retriever
adds the filter to its own SQL, and dense search uses pgvector's iterative
index scan to keep walking the HNSW graph until enough rows pass the filter.
"""

from dataclasses import dataclass, field
from typing import Any

from psycopg import sql


@dataclass(frozen=True)
class Filters:
    year_from: int | None = None
    year_to: int | None = None
    venues: tuple[str, ...] = field(default_factory=tuple)  # normalized names, e.g. "ICLR"
    categories: tuple[str, ...] = field(default_factory=tuple)  # arXiv, e.g. "cs.CL"
    # The venue's edition ("ICLR 2024"), not the arXiv date: an ICLR 2025 paper
    # is often posted in 2024.
    venue_year: int | None = None

    @property
    def empty(self) -> bool:
        return not (
            self.year_from or self.year_to or self.venues or self.categories or self.venue_year
        )


def filter_sql(f: Filters, paper: str = "p") -> tuple[sql.Composable, dict[str, Any]]:
    """A boolean SQL condition on the papers row aliased `paper`, plus its params.

    Years compare on the v1 publication date. Venue matches papers.venue (only
    papers with a detected venue pass). Category matches any listed category.
    """
    alias = sql.Identifier(paper)
    parts: list[sql.Composable] = []
    params: dict[str, Any] = {}
    if f.year_from:
        parts.append(sql.SQL("{}.published >= make_date(%(f_year_from)s, 1, 1)").format(alias))
        params["f_year_from"] = int(f.year_from)
    if f.year_to:
        parts.append(sql.SQL("{}.published < make_date(%(f_year_to)s + 1, 1, 1)").format(alias))
        params["f_year_to"] = int(f.year_to)
    if f.venues:
        parts.append(sql.SQL("{}.venue = ANY(%(f_venues)s)").format(alias))
        params["f_venues"] = list(f.venues)
    if f.venue_year:
        parts.append(sql.SQL("{}.venue_year = %(f_venue_year)s").format(alias))
        params["f_venue_year"] = int(f.venue_year)
    if f.categories:
        parts.append(
            sql.SQL(
                "EXISTS (SELECT 1 FROM paper_categories c "
                "WHERE c.paper_id = {}.id AND c.category = ANY(%(f_categories)s))"
            ).format(alias)
        )
        params["f_categories"] = list(f.categories)
    if not parts:
        return sql.SQL("TRUE"), params
    return sql.SQL(" AND ").join(parts), params
