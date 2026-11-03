from papergraph.api.main import make_filters
from papergraph.retrieval.filters import Filters, filter_sql


def test_empty_filters_are_true() -> None:
    cond, params = filter_sql(Filters())
    assert cond.as_string(None) == "TRUE"
    assert params == {}
    assert Filters().empty


def test_all_filters_render_as_parameters() -> None:
    cond, params = filter_sql(Filters(2020, 2023, ("ICLR",), ("cs.CL",)), "pp")
    text = cond.as_string(None)
    assert '"pp".published >= make_date(%(f_year_from)s, 1, 1)' in text
    assert '"pp".published < make_date(%(f_year_to)s + 1, 1, 1)' in text  # inclusive year_to
    assert '"pp".venue = ANY(%(f_venues)s)' in text
    assert "c.category = ANY(%(f_categories)s)" in text
    # Values travel as bind parameters, never inside the SQL text.
    assert params == {
        "f_year_from": 2020,
        "f_year_to": 2023,
        "f_venues": ["ICLR"],
        "f_categories": ["cs.CL"],
    }


def test_api_parses_comma_separated_filters() -> None:
    f = make_filters(2021, None, "ICLR, NeurIPS,", "")
    assert f == Filters(2021, None, ("ICLR", "NeurIPS"), ())
