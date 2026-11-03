import pytest

from papergraph.search import name_matches


@pytest.mark.parametrize(
    "candidate",
    ["Yann LeCun", "Yann Lecun", "Y. LeCun", "Yann André LeCun", "YANN LECUN", "Y LeCun"],
)
def test_spelling_variants_match(candidate: str) -> None:
    assert name_matches("Yann LeCun", candidate)


@pytest.mark.parametrize("candidate", ["Le Cam", "Xavier LeCun", "Yann Dauphin", "LeCunha Y."])
def test_other_people_do_not_match(candidate: str) -> None:
    assert not name_matches("Yann LeCun", candidate)


def test_initial_in_the_query() -> None:
    assert name_matches("Y. Bengio", "Yoshua Bengio")
    assert not name_matches("S. Bengio", "Yoshua Bengio")


def test_surname_only_matches_any_first_name() -> None:
    assert name_matches("Bengio", "Yoshua Bengio")
    assert name_matches("Bengio", "Samy Bengio")


def test_empty_names() -> None:
    assert not name_matches("", "Yann LeCun")
    assert not name_matches("Yann LeCun", "")


def test_surname_first_order() -> None:
    assert name_matches("Fei-Fei Li", "Li Fei-Fei")
    assert name_matches("Fei-Fei Li", "Fei-Fei Li")
    assert not name_matches("Fei-Fei Li", "Li Wei")
