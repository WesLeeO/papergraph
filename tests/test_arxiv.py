import json
from datetime import date
from pathlib import Path
from typing import Any

from papergraph.ingest.arxiv import clean_text, iter_papers, parse_record


def record(**overrides: Any) -> dict[str, Any]:
    """A raw snapshot record shaped like the real file."""
    base: dict[str, Any] = {
        "id": "1706.03762",
        "title": "Attention Is All\n  You Need",
        "abstract": "  The dominant sequence transduction models\nare based on ...\n",
        "doi": None,
        "categories": "cs.CL cs.LG",
        "comments": "15 pages, 5 figures",
        "journal-ref": None,
        "authors_parsed": [["Vaswani", "Ashish", ""], ["Shazeer", "Noam", ""]],
        "versions": [
            {"version": "v1", "created": "Mon, 12 Jun 2017 17:57:34 GMT"},
            {"version": "v2", "created": "Mon, 19 Jun 2017 16:49:45 GMT"},
        ],
        "update_date": "2023-08-03",
    }
    return base | overrides


def test_parses_ml_paper() -> None:
    paper = parse_record(record())
    assert paper is not None
    assert paper.title == "Attention Is All You Need"
    assert paper.abstract == "The dominant sequence transduction models are based on ..."
    assert paper.categories == ["cs.CL", "cs.LG"]
    assert paper.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert paper.published == date(2017, 6, 12)  # v1, not the latest version
    assert paper.updated == date(2023, 8, 3)
    assert paper.venue is None


def test_skips_non_ml_paper() -> None:
    assert parse_record(record(categories="hep-ph")) is None


def test_cross_listed_paper_is_kept() -> None:
    paper = parse_record(record(categories="math.OC cs.LG"))
    assert paper is not None
    assert paper.categories[0] == "math.OC"  # primary category stays first


def test_skips_record_without_versions() -> None:
    assert parse_record(record(versions=[])) is None


def test_extracts_venue_from_comments() -> None:
    paper = parse_record(record(comments="Accepted at NIPS 2017"))
    assert paper is not None and paper.venue is not None
    assert (paper.venue.venue, paper.venue.year) == ("NeurIPS", 2017)


def test_author_suffix_is_kept() -> None:
    paper = parse_record(record(authors_parsed=[["King", "Martin Luther", "Jr"]]))
    assert paper is not None
    assert paper.authors == ["Martin Luther King Jr"]


def test_clean_text() -> None:
    assert clean_text("  a\n  b\tc  ") == "a b c"


def test_iter_papers_from_year_keeps_recent_papers(tmp_path: Path) -> None:
    def v1(created: str) -> list[dict[str, str]]:
        return [{"version": "v1", "created": created}]

    rows = [
        record(id="old", versions=v1("Mon, 12 Jun 2017 17:57:34 GMT")),
        record(id="physics", categories="hep-ph", versions=v1("Tue, 03 Feb 2026 10:00:00 GMT")),
        record(id="new", versions=v1("Tue, 03 Feb 2026 10:00:00 GMT")),
    ]
    path = tmp_path / "snapshot.json"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert [p.id for p in iter_papers(path)] == ["old", "new"]
    assert [p.id for p in iter_papers(path, from_year=2026)] == ["new"]
