"""Edge cases against the real database and models.

Skipped by default (CI has neither). Run after `make setup`:

    PAPERGRAPH_LIVE=1 uv run --group embed pytest tests/integration
"""

import os
import time
from collections.abc import Iterator

import psycopg
import pytest

from papergraph import search as svc
from papergraph.db import connect
from papergraph.retrieval.filters import Filters

pytestmark = pytest.mark.skipif(
    os.environ.get("PAPERGRAPH_LIVE") != "1", reason="needs the database: set PAPERGRAPH_LIVE=1"
)


@pytest.fixture(scope="module")
def conn() -> Iterator[psycopg.Connection]:
    with connect("public", autocommit=True) as c:
        yield c


ODD_QUERIES = [
    "'; DROP TABLE papers; --",  # SQL injection attempt
    "a & b | !c <-> (d",  # tsquery operators
    "((((((",
    "***",
    "the of and in to",  # stop words only: no keyword terms at all
    "🤖 robots learning to walk 🦾",
    "图神经网络",  # Chinese ("graph neural networks")
    "réseaux de neurones convolutifs",  # French
    "2017",
    "x" * 500,  # maximum length
    "LoRA",  # short acronym
    "\\x00 \\n \\t",
]


@pytest.mark.parametrize("q", ODD_QUERIES)
def test_odd_queries_never_crash(conn: psycopg.Connection, q: str) -> None:
    papers = svc.search(conn, q, k=10)
    assert isinstance(papers, list)
    assert len({p.id for p in papers}) == len(papers)  # no duplicates


def test_database_survives_the_injection_attempt(conn: psycopg.Connection) -> None:
    svc.search(conn, "'; DROP TABLE papers; --", k=1)
    row = conn.execute("SELECT count(*) FROM papers").fetchone()
    assert row is not None and row[0] > 0


def test_stop_word_query_still_returns_dense_results(conn: psycopg.Connection) -> None:
    # BM25 finds no terms; dense retrieval must still answer.
    assert len(svc.search(conn, "the of and in to", k=5)) == 5


def test_pages_are_consistent_slices_of_one_ranking(conn: psycopg.Connection) -> None:
    q = "graph neural networks for molecules"
    page1, page2 = svc.search(conn, q, k=20), svc.search(conn, q, k=20, offset=20)
    ids = [p.id for p in page1 + page2]
    assert ids == svc.ranked_ids(conn, q)[:40]
    assert len(set(ids)) == 40


def test_cached_next_page_is_fast(conn: psycopg.Connection) -> None:
    q = "speculative decoding for llm inference"
    svc.ranked_ids(conn, q)
    t = time.perf_counter()
    svc.search(conn, q, k=20, offset=100)
    assert time.perf_counter() - t < 0.5


def test_unknown_venue_filter_returns_nothing(conn: psycopg.Connection) -> None:
    assert svc.search(conn, "transformers", k=10, filters=Filters(venues=("NotAVenue",))) == []


def test_year_filter_is_respected(conn: psycopg.Connection) -> None:
    papers = svc.search(conn, "diffusion models", k=30, filters=Filters(2020, 2021))
    assert papers and all(p.published and "2020" <= p.published[:4] <= "2021" for p in papers)


def test_venue_filter_is_respected(conn: psycopg.Connection) -> None:
    papers = svc.search(conn, "image segmentation", k=30, filters=Filters(venues=("CVPR",)))
    assert papers and all(p.venue == "CVPR" for p in papers)


def test_category_filter_is_respected(conn: psycopg.Connection) -> None:
    papers = svc.search(conn, "speech recognition", k=20, filters=Filters(categories=("cs.CV",)))
    assert papers and all("cs.CV" in p.categories for p in papers)


def test_empty_year_range_returns_nothing(conn: psycopg.Connection) -> None:
    assert svc.search(conn, "transformers", k=10, filters=Filters(1991, 1991)) == []


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Yann LeCun", "Yann LeCun"),
        ("yann lecun", "Yann LeCun"),  # case
        ("Y. LeCun", "Yann LeCun"),  # initial
        ("LeCun", "Yann LeCun"),  # surname only
        ("Fei-Fei Li", "Li Fei-Fei"),  # surname-first spelling in arXiv
        ("Bernhard Schölkopf", "Bernhard Schölkopf"),  # accents
    ],
)
def test_author_name_variants(conn: psycopg.Connection, name: str, expected: str) -> None:
    assert expected in [n for n, _ in svc.find_authors(conn, name)]


@pytest.mark.parametrize("name", ["%", "_", "a", "%%%%", "Nobody Atall", "'; DROP TABLE x; --"])
def test_hostile_or_unknown_author_names_match_nothing_quickly(
    conn: psycopg.Connection, name: str
) -> None:
    t = time.perf_counter()
    assert svc.find_authors(conn, name) == []
    assert time.perf_counter() - t < 1.0


def test_author_papers_are_all_by_that_author(conn: psycopg.Connection) -> None:
    variants, papers = svc.author_papers(conn, "Kaiming He", k=50)
    names = {n for n, _ in variants}
    assert papers and all(names & set(p.authors) for p in papers)


def test_author_papers_newest_first(conn: psycopg.Connection) -> None:
    _, papers = svc.author_papers(conn, "Yann LeCun", k=30)
    dates = [p.published for p in papers if p.published]
    assert dates == sorted(dates, reverse=True)


def test_author_with_topic_puts_the_topic_first(conn: psycopg.Connection) -> None:
    _, papers = svc.author_papers(conn, "Kaiming He", query="masked autoencoders", k=3)
    assert "masked autoencoder" in papers[0].title.lower()


def test_similar_papers_exclude_the_paper_itself(conn: psycopg.Connection) -> None:
    pid = svc.search(conn, "masked autoencoders vision", k=1)[0].id
    similar = svc.similar_papers(conn, pid, k=10)
    assert len(similar) == 10 and pid not in {p.id for p in similar}


def test_unknown_paper_id(conn: psycopg.Connection) -> None:
    assert svc.get_paper(conn, "9999.99999") is None
    assert svc.similar_papers(conn, "9999.99999") == []


# --- query understanding on real data -------------------------------------------------


@pytest.mark.parametrize("text", ["Yann Le Cun", "yann lecun", "Y. LeCun", "papers by Yann LeCun"])
def test_typing_a_name_lists_that_authors_papers(conn: psycopg.Connection, text: str) -> None:
    r = svc.run_query(conn, text)
    assert r.author_name and r.understood.topic == ""
    names = {n for n, _ in r.matched_authors}
    papers = svc.fetch_papers(conn, r.ids[:20])
    assert len(papers) == 20 and all(names & set(p.authors) for p in papers)


def test_name_topic_and_year_together(conn: psycopg.Connection) -> None:
    r = svc.run_query(conn, "Yann Le Cun world models since 2024")
    assert r.understood.topic == "world models" and r.filters.year_from == 2024
    papers = svc.fetch_papers(conn, r.ids[:10])
    assert papers and all(p.published and p.published >= "2024" for p in papers)
    assert any("world model" in p.title.lower() for p in papers[:3])


def test_venue_and_year_without_topic_browses_newest_first(conn: psycopg.Connection) -> None:
    r = svc.run_query(conn, "ICLR 2024")
    papers = svc.fetch_papers(conn, r.ids[:20])
    assert papers and all(p.venue == "ICLR" and p.venue_year == 2024 for p in papers)


def test_arxiv_id_goes_straight_to_the_paper(conn: psycopg.Connection) -> None:
    r = svc.run_query(conn, "arXiv:2111.06377v3")
    assert r.ids[0] == "2111.06377"


@pytest.mark.parametrize(
    "text",
    [
        "Graph Attention Networks",
        "Large Language Models",
        "Deep Reinforcement Learning",
        "Neural Radiance Fields",
        "Vision Transformers",
        "Kim Possible robots",  # a common surname in a topic
    ],
)
def test_topic_phrases_are_not_mistaken_for_authors(conn: psycopg.Connection, text: str) -> None:
    r = svc.run_query(conn, text)
    assert r.author_name is None and r.understood.topic == text


def test_removing_the_author_chip_searches_the_name_as_text(conn: psycopg.Connection) -> None:
    r = svc.run_query(conn, "Yann LeCun", skip=["author"])
    assert r.author_name is None and r.understood.topic == "Yann LeCun"
