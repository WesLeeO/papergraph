from datetime import date

from papergraph.evals.benchmarks.litsearch import (
    assign_splits,
    paper_row,
    parse_query,
    query_id,
)

ROW = {
    "query_set": "manual_iclr",
    "query": "  Papers on   sparse attention\nfor long documents ",
    "specificity": 0,
    "corpusids": [123, 456],
}


def test_parse_query_cleans_text_and_maps_fields() -> None:
    q = parse_query(ROW)
    assert q.text == "Papers on sparse attention for long documents"
    assert q.qid == query_id(q.text)
    assert q.relevant == ("123", "456")  # ids are text, like papers.id
    assert q.specificity == "broad"
    assert q.stratum == "manual_iclr/broad"


def test_query_id_is_stable() -> None:
    # Changing how ids are derived would orphan the committed split file.
    assert query_id("hello") == "aaf4c61ddcc5"


def _strata(n_a: int, n_b: int) -> dict[str, str]:
    return {query_id(f"a{i}"): "A" for i in range(n_a)} | {
        query_id(f"b{i}"): "B" for i in range(n_b)
    }


def test_split_is_stratified() -> None:
    strata = _strata(30, 9)
    splits = assign_splits(strata, dev_fraction=1 / 3)
    for stratum, expected_dev in [("A", 10), ("B", 3)]:
        dev = [q for q, s in strata.items() if s == stratum and splits[q] == "dev"]
        assert len(dev) == expected_dev


def test_split_ignores_input_order() -> None:
    strata = _strata(30, 9)
    shuffled = dict(reversed(list(strata.items())))
    assert assign_splits(strata) == assign_splits(shuffled)


def test_split_covers_every_query_once() -> None:
    strata = _strata(7, 5)
    splits = assign_splits(strata)
    assert splits.keys() == strata.keys()
    assert set(splits.values()) <= {"dev", "test"}


def test_paper_row_uses_year_and_cleans_text() -> None:
    row = paper_row(42, "A\n  Title", None, 2021.0, "")
    assert row == ("42", "A Title", "", date(2021, 1, 1), None)


def test_paper_row_without_year() -> None:
    assert paper_row(42, "T", "abs", None, "10.1/x")[3] is None
