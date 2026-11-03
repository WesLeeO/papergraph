"""Parse the arXiv metadata snapshot (JSON Lines) into clean `Paper` records."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from papergraph.ingest.venues import VenueMatch, extract_venue

ML_CATEGORIES = frozenset({"cs.LG", "stat.ML", "cs.CL", "cs.CV", "cs.NE", "cs.IR"})


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    abstract: str
    doi: str | None
    categories: list[str]  # first one is the primary category
    authors: list[str]  # "First Last", in author order
    published: date  # date of v1
    updated: date | None
    venue: VenueMatch | None


def clean_text(text: str) -> str:
    """Collapse arXiv's hard line breaks and indentation into single spaces."""
    return " ".join(text.split())


def _author_name(parts: list[str]) -> str:
    # authors_parsed entries are [last, first, suffix], already LaTeX-decoded.
    last, first, *rest = [*parts, "", ""]
    suffix = rest[0] if rest else ""
    return clean_text(f"{first} {last} {suffix}")


def parse_record(raw: dict[str, Any]) -> Paper | None:
    """A Paper if the record is an ML paper with the fields we need, else None."""
    categories = raw.get("categories", "").split()
    if not ML_CATEGORIES.intersection(categories):
        return None
    title = clean_text(raw.get("title") or "")
    versions = raw.get("versions") or []
    if not title or not versions:
        return None
    published = parsedate_to_datetime(versions[0]["created"]).date()
    updated = date.fromisoformat(raw["update_date"]) if raw.get("update_date") else None
    return Paper(
        id=raw["id"],
        title=title,
        abstract=clean_text(raw.get("abstract") or ""),
        doi=raw.get("doi") or None,
        categories=categories,
        authors=[a for p in raw.get("authors_parsed") or [] if (a := _author_name(p))],
        published=published,
        updated=updated,
        venue=extract_venue(raw.get("comments"), raw.get("journal-ref")),
    )


def iter_papers(
    path: Path, limit: int | None = None, from_year: int | None = None
) -> Iterator[Paper]:
    """Stream ML papers from the snapshot, one line at a time (the file is ~5 GB).

    from_year keeps only papers first published in that year or later: a quick
    way to a small, recent corpus (the snapshot itself is ordered oldest first,
    so `limit` alone would give the oldest papers).
    """
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            paper = parse_record(json.loads(line))
            if paper is None:
                continue
            if from_year is not None and paper.published.year < from_year:
                continue
            yield paper
            count += 1
            if limit is not None and count >= limit:
                return
