"""Query understanding, with a fake author index (no database)."""

from datetime import date

import pytest

from papergraph.query import understand
from papergraph.search import name_matches

AUTHORS = ["Yann LeCun", "Li Fei-Fei", "Kaiming He", "Laurens van der Maaten", "Geoffrey Hinton"]
VENUES = ["NeurIPS", "ICML", "ICLR", "CVPR", "EMNLP", "EMNLP Findings", "ACL", "ACL Findings"]


def parse(text: str, skip: tuple[str, ...] = ()) -> dict[str, object]:
    u = understand(
        text,
        venues=VENUES,
        is_author=lambda name: any(name_matches(name, a) for a in AUTHORS),
        skip=skip,
        today=date(2026, 11, 3),
    )
    return {
        k: v
        for k, v in {
            "topic": u.topic,
            "author": u.author,
            "venues": u.venues,
            "year_from": u.year_from,
            "year_to": u.year_to,
            "venue_year": u.venue_year,
            "arxiv_id": u.arxiv_id,
        }.items()
        if v not in (None, ())
    }


def test_plain_topic_is_untouched() -> None:
    assert parse("reducing hallucination in retrieval-augmented generation") == {
        "topic": "reducing hallucination in retrieval-augmented generation"
    }


@pytest.mark.parametrize(
    "text", ["Yann Le Cun", "yann lecun", "Yann LeCun", "papers by Yann LeCun"]
)
def test_bare_name_is_an_author(text: str) -> None:
    assert parse(text)["author"] is not None
    assert parse(text)["topic"] == ""


def test_the_full_example() -> None:
    assert parse("Yann Le Cun world models since 2024 at ICLR") == {
        "topic": "world models",
        "author": "Yann Le Cun",
        "venues": ("ICLR",),
        "year_from": 2024,
    }


def test_author_after_by_inside_a_topic_query() -> None:
    assert parse("masked autoencoders by kaiming he") == {
        "topic": "masked autoencoders",
        "author": "kaiming he",
    }


def test_capitalized_topic_words_are_not_authors() -> None:
    # Looks like a name (two capitalized words), but no such author exists.
    assert parse("Graph Attention Networks") == {"topic": "Graph Attention Networks"}


@pytest.mark.parametrize(
    ("text", "years"),
    [
        ("diffusion models since 2022", {"year_from": 2022}),
        ("diffusion models after 2022", {"year_from": 2023}),
        ("diffusion models before 2020", {"year_to": 2019}),
        ("diffusion models until 2020", {"year_to": 2020}),
        ("diffusion models in 2023", {"year_from": 2023, "year_to": 2023}),
        ("diffusion models 2019-2021", {"year_from": 2019, "year_to": 2021}),
        ("diffusion models from 2021 to 2019", {"year_from": 2019, "year_to": 2021}),
        ("recent diffusion models", {"year_from": 2025}),
        ("diffusion models since 2020 before 2023", {"year_from": 2020, "year_to": 2022}),
    ],
)
def test_year_phrases(text: str, years: dict[str, int]) -> None:
    got = parse(text)
    assert got["topic"] == "diffusion models"
    assert {k: got[k] for k in ("year_from", "year_to") if k in got} == years


def test_bare_year_in_a_topic_stays_in_the_topic() -> None:
    # "COCO 2017" is a dataset name, not a date filter.
    assert parse("object detection on COCO 2017") == {"topic": "object detection on COCO 2017"}


@pytest.mark.parametrize(
    ("text", "venue", "year"),
    [
        ("ICLR papers on diffusion", "ICLR", None),
        ("diffusion at iclr 2024", "ICLR", 2024),
        ("diffusion NeurIPS'23", "NeurIPS", 2023),
        ("diffusion nips", "NeurIPS", None),
        ("diffusion EMNLP Findings", "EMNLP Findings", None),
    ],
)
def test_venues(text: str, venue: str, year: int | None) -> None:
    got = parse(text)
    assert got["venues"] == (venue,) and got["topic"] == "diffusion"
    assert got.get("venue_year") == year  # the edition, not a date filter
    assert "year_from" not in got


@pytest.mark.parametrize(
    ("text", "arxiv_id"),
    [
        ("2310.06825", "2310.06825"),
        ("arXiv:2310.06825v2", "2310.06825"),
        ("cs/0112017", "cs/0112017"),
    ],
)
def test_arxiv_ids(text: str, arxiv_id: str) -> None:
    assert parse(text)["arxiv_id"] == arxiv_id


def test_skip_leaves_the_text_in_the_topic() -> None:
    # The user removed the Author chip: search the name as a topic instead.
    assert parse("Yann LeCun", skip=("author",)) == {"topic": "Yann LeCun"}
    got = parse("diffusion at ICLR since 2023", skip=("venue",))
    assert "venues" not in got and got["year_from"] == 2023 and "ICLR" in str(got["topic"])


def test_filters_only_query_has_an_empty_topic() -> None:
    expected = {"topic": "", "venues": ("ICLR",), "venue_year": 2024}
    assert parse("ICLR 2024") == expected


@pytest.mark.parametrize("text", ["", "   ", "the", "!!!", "by", "since"])
def test_degenerate_input(text: str) -> None:
    got = parse(text)
    assert set(got) <= {"topic"}


def test_venue_year_and_date_range_together() -> None:
    got = parse("diffusion at NeurIPS 2023 since 2022")
    assert got["venue_year"] == 2023 and got["year_from"] == 2022
