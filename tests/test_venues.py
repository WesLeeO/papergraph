import pytest

from papergraph.ingest.venues import extract_venue


@pytest.mark.parametrize(
    ("comments", "venue", "year"),
    [
        # Real arXiv comment styles.
        ("To appear at ICLR 2014", "ICLR", 2014),
        ("Accepted in CVPR'19. Code is available at https://github.com/x", "CVPR", 2019),
        ("published in ICML-2019", "ICML", 2019),
        ("Appearing in Neural Information Processing Systems (NIPS) 2010", "NeurIPS", 2010),
        (
            "Accepted by the 36th International Conference on Machine Learning (ICML 2019)",
            "ICML",
            2019,
        ),
        ("NeurIPS 2023, 21 pages, 20 figures", "NeurIPS", 2023),
        ("ACL 2019 camera ready; code in https://github.com/x", "ACL", 2019),
        ("Accepted to Findings of EMNLP 2023", "EMNLP Findings", 2023),
        ("in IEEE CVPR 2015, Boston, MA, June 2015", "CVPR", 2015),
        ("Published in Nature Machine Intelligence", "Nature Machine Intelligence", None),
    ],
)
def test_extracts_accepted_venues(comments: str, venue: str, year: int | None) -> None:
    match = extract_venue(comments, None)
    assert match is not None
    assert (match.venue, match.year) == (venue, year)


@pytest.mark.parametrize(
    "comments",
    [
        "Submitted to CVPR 2017 (10 pages, 5 figures)",
        "Under review at ICLR 2024",
        "Accepted at the ICLR 2019 SafeML Workshop",
        "CVPR-2021 HACS Challenge - Weakly-supervised Learning Track champion",
        "We compare against ICLR baselines",  # conference without a year
        "Published in Nature Communications",  # different journal
        "12 pages, 5 figures",
        # Regressions found by spot-checking real ingests:
        "Proceedings of the Conference on Parallel Problem Solving from Nature",
        "Advances in Neural Information Processing Systems 23 (NIPS), 2038-2046",
        "Published at SDProc @ ACL 2024",
        "Accepted At ICLR 2024 (Tiny Paper Track)",
    ],
)
def test_rejects_uncertain_mentions(comments: str) -> None:
    assert extract_venue(comments, None) is None


def test_volume_number_is_not_a_year() -> None:
    match = extract_venue(None, "Journal of Machine Learning Research, 13(Aug): 2237--2278, 2012")
    assert match is not None
    assert (match.venue, match.year) == ("JMLR", None)


def test_old_two_digit_year() -> None:
    match = extract_venue("Appeared at NIPS'98", None)
    assert match is not None
    assert match.year == 1998


def test_journal_ref_wins_over_comments() -> None:
    match = extract_venue("Submitted to NeurIPS 2020", "ICML 2021")
    assert match is not None
    assert match.venue == "ICML"


def test_earliest_mention_wins() -> None:
    match = extract_venue("ICML 2021; extended version of our AAAI 2020 paper", None)
    assert match is not None
    assert match.venue == "ICML"


def test_line_breaks_are_collapsed() -> None:
    comments = "Accepted by the 36th International Conference on Machine\n  Learning (ICML 2019)"
    match = extract_venue(comments, None)
    assert match is not None
    assert match.venue == "ICML"
