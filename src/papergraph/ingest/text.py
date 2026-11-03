"""What text represents a paper for embedding."""


def paper_text(title: str, abstract: str) -> str:
    """The string that gets embedded for one paper: title, newline, abstract.

    - Title first: if a long abstract exceeds bge-small's 512-token limit, only
      the end of the abstract is truncated, never the title.
    - Newline separator: keeps the title a distinct unit instead of merging its
      last word into the abstract's first ("NeedThe").
    - Empty parts are dropped, so an empty abstract gives just the title.
    """
    return "\n".join(part for part in (title.strip(), abstract.strip()) if part)
