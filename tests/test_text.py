"""Contract for paper_text. These fail until you implement it."""

from papergraph.ingest.text import paper_text

TITLE = "Attention Is All You Need"
ABSTRACT = "The dominant sequence transduction models are based on recurrent networks."


def test_title_is_included() -> None:
    assert TITLE in paper_text(TITLE, ABSTRACT)


def test_empty_abstract_gives_clean_text() -> None:
    text = paper_text(TITLE, "")
    assert text.strip() == text, "no leading/trailing whitespace or separators"
    assert TITLE in text


def test_title_and_abstract_stay_separate_words() -> None:
    # Regression: plain concatenation produced "NeedThe".
    assert "NeedThe" not in paper_text(TITLE, ABSTRACT)


def test_is_deterministic() -> None:
    # Embeddings are cached by input hash: the same paper must give the same text.
    assert paper_text(TITLE, ABSTRACT) == paper_text(TITLE, ABSTRACT)
