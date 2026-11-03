"""Query understanding: split what a user typed into filters and a topic.

    "Yann Le Cun world models since 2024 at ICLR"
      -> author "Yann Le Cun", venue ICLR, year >= 2024, topic "world models"

Only the topic goes to retrieval (BM25 + embeddings match paper CONTENT);
names, venues, years and arXiv ids become filters. Everything here is rules
and lookups: fast, deterministic, testable. The UI shows each understood part
as a removable chip, and removing one re-runs the query with that kind of
parsing skipped (`skip`), so a wrong guess costs one click.

The author check needs the database, so it is passed in (`is_author`): this
module stays pure and is unit-tested without Postgres.
"""

import re
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field, replace
from datetime import date

KINDS = ("id", "author", "venue", "year")

# Extra spellings people type for venues we normalize (lowercase -> canonical).
VENUE_ALIASES = {"nips": "NeurIPS", "neurips": "NeurIPS"}

# Words that only frame the request ("papers by X since 2020"): dropped from the
# topic once something was understood. Plain topic queries are never touched.
FILLER = frozenset(
    {
        "papers", "paper", "articles", "article", "publications", "publication", "works",
        "work", "research", "results", "show", "me", "find", "list", "all", "any", "some",
        "what", "which", "has", "have", "published", "written", "authored", "by", "from",
        "at", "in", "on", "about", "of", "the", "a", "an", "and", "recent", "latest", "new",
        "newest", "recently", "please",
    }
)  # fmt: skip

# Parts of surnames that look like stop words ("van der Maaten", "Le Cun").
NAME_PARTICLES = frozenset(
    {"van", "der", "von", "de", "la", "le", "di", "da", "del", "du", "dos", "ten", "ter"}
)
NOT_NAME = FILLER - NAME_PARTICLES | {"for", "with", "to", "vs", "via", "using"}

APOSTROPHES = "'\u2019"  # straight and typographic
DASHES = "-\u2013"  # hyphen and en dash

_ARXIV_ID = re.compile(
    r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b|\b([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?\b"
)
_YEAR = r"((?:19|20)\d{2})"
_YEAR_RANGE = re.compile(
    rf"\b(?:from\s+|between\s+)?{_YEAR}\s*(?:[{DASHES}]|to|until|and)\s*{_YEAR}\b", re.I
)
_YEAR_FROM = re.compile(rf"\b(since|after|from)\s+{_YEAR}\b", re.I)
_YEAR_TO = re.compile(rf"\b(before|until|up to)\s+{_YEAR}\b", re.I)
_YEAR_IN = re.compile(rf"\b(?:in|during)\s+{_YEAR}\b", re.I)
_RECENT = re.compile(r"\b(recent|latest|newest|recently)\b", re.I)
_BY = re.compile(r"\bby\s+", re.I)
_WORD = re.compile(rf"[\w.\-{APOSTROPHES}]+", re.UNICODE)


@dataclass(frozen=True)
class Understood:
    """What a query means. Each filter keeps the text it came from, for chips."""

    topic: str
    author: str | None = None  # the name as typed; resolved to spellings later
    venues: tuple[str, ...] = ()
    venue_year: int | None = None  # "ICLR 2024": the edition, not the arXiv date
    year_from: int | None = None
    year_to: int | None = None
    arxiv_id: str | None = None
    spans: dict[str, str] = field(default_factory=dict)  # kind -> text consumed

    @property
    def structured(self) -> bool:
        return bool(
            self.author
            or self.venues
            or self.venue_year
            or self.year_from
            or self.year_to
            or self.arxiv_id
        )


def _cut(text: str, start: int, end: int) -> str:
    return (text[:start] + " " + text[end:]).strip()


def _clean_topic(text: str) -> str:
    words = [w for w in _WORD.findall(text) if w.lower().strip("." + APOSTROPHES) not in FILLER]
    topic = " ".join(words).strip(" .,-" + APOSTROPHES)
    return topic if any(ch.isalnum() for ch in topic) else ""


def _looks_like_name(words: Sequence[str], whole_query: bool) -> bool:
    """Cheap pre-check before a database lookup: 2-4 words, no filler words,
    and capitalized (or the whole query, where people often type lowercase)."""
    if not 2 <= len(words) <= 4:
        return False
    if any(w.lower() in NOT_NAME or not any(ch.isalpha() for ch in w) for w in words):
        return False
    if whole_query:
        return True
    return all(w[0].isupper() or w.lower() in NAME_PARTICLES for w in words)


def understand(
    text: str,
    *,
    venues: Collection[str],
    is_author: Callable[[str], bool],
    skip: Collection[str] = (),
    today: date | None = None,
) -> Understood:
    """Parse `text`. `venues`: known venue names; `is_author(name)`: whether a
    name refers to a known author; `skip`: kinds not to parse (from KINDS)."""
    original = " ".join(text.split())
    rest = original
    found: dict[str, object] = {}
    spans: dict[str, str] = {}

    # arXiv id: "2310.06825", "arXiv:2310.06825v2", "cs/0112017"
    if "id" not in skip and (m := _ARXIV_ID.search(rest)):
        found["arxiv_id"] = m.group(1) or m.group(2)
        spans["id"] = m.group(0)
        rest = _cut(rest, m.start(), m.end())

    # Venue, optionally with a year: "ICLR", "iclr 2024", "NeurIPS'23"
    if "venue" not in skip:
        names = {v.lower(): v for v in venues} | VENUE_ALIASES
        for low in sorted(names, key=len, reverse=True):  # "EMNLP Findings" before "EMNLP"
            pattern = rf"\b{re.escape(low)}\b(?:\s*[{APOSTROPHES}](\d{{2}})\b|\s+{_YEAR}\b)?"
            m = re.search(pattern, rest, re.I)
            if m:
                found["venues"] = (names[low],)
                spans["venue"] = m.group(0)
                year = m.group(2) or (f"20{m.group(1)}" if m.group(1) else None)
                if year:
                    found["venue_year"] = int(year)
                rest = _cut(rest, m.start(), m.end())
                break

    # Years: ranges, "since/after", "before/until", "in 2023", "recent"
    if "year" not in skip:
        this_year = (today or date.today()).year
        if m := _YEAR_RANGE.search(rest):
            a, b = sorted((int(m.group(1)), int(m.group(2))))
            found["year_from"], found["year_to"] = a, b
        elif m := _YEAR_IN.search(rest):
            found["year_from"] = found["year_to"] = int(m.group(1))
        else:
            if m := _YEAR_FROM.search(rest):
                y = int(m.group(2))
                found["year_from"] = y + 1 if m.group(1).lower() == "after" else y
            if m2 := _YEAR_TO.search(rest):
                y = int(m2.group(2))
                found["year_to"] = y - 1 if m2.group(1).lower() == "before" else y
                spans["year"] = m2.group(0)
                rest = _cut(rest, m2.start(), m2.end())
            if m is None and m2 is None and (m := _RECENT.search(rest)):
                found["year_from"] = this_year - 1
        if m is not None and ("year_from" in found or "year_to" in found):
            spans["year"] = (m.group(0) + " " + spans.get("year", "")).strip()
            rest = _cut(rest, m.start(), m.end())

    # Author: after "by", or a capitalized 2-4 word run (a whole short query
    # may be lowercase). Longest and leftmost runs are tried first.
    if "author" not in skip:
        words = _WORD.findall(rest)
        candidates: list[tuple[int, int]] = []
        if by := _BY.search(rest):
            after = len(_WORD.findall(rest[: by.end()]))
            candidates += [(after, n) for n in range(min(4, len(words) - after), 0, -1)]
        whole = len(words) <= 4
        for n in (4, 3, 2):
            candidates += [(i, n) for i in range(len(words) - n + 1)]
        for i, n in candidates:
            run = words[i : i + n]
            after_by = by is not None and i == len(_WORD.findall(rest[: by.end()]))
            plausible = _looks_like_name(run, whole and n == len(words)) or (
                after_by and all(w.lower() not in NOT_NAME for w in run)
            )
            if plausible and is_author(" ".join(run)):
                name = " ".join(run)
                found["author"] = name
                spans["author"] = name
                rest = re.sub(rf"(?:\bby\s+)?{re.escape(name)}", " ", rest, count=1, flags=re.I)
                break

    result = Understood(topic=original, spans=spans, **found)  # type: ignore[arg-type]
    if not result.structured:
        return result  # a plain topic query: leave it exactly as typed
    return replace(result, topic=_clean_topic(rest))
