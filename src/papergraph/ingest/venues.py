"""Extract a publication venue from arXiv's free-text `comments` / `journal-ref`.

Precision over recall: a wrong venue in a filter is worse than a missing one, so
anything that looks like a submission, a workshop or a challenge is skipped.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class VenueMatch:
    venue: str  # normalized name, e.g. "NeurIPS"
    year: int | None
    raw: str  # the text the match came from


# Normalized name -> regex alternatives. Longer, more specific names come first
# ("Nature Machine Intelligence" before "Nature").
_ALIASES: dict[str, list[str]] = {
    "Nature Machine Intelligence": [r"Nature Machine Intelligence"],
    # Only "in Nature" / "to Nature" / a journal-ref starting with "Nature": not
    # "Problem Solving from Nature" (a conference) or "Nature Communications".
    "Nature": [r"(?:^|(?<=\bin )|(?<=\bto )|(?<=\bby )|(?<=\bat ))Nature(?!\s+[A-Z])"],
    "NeurIPS": [r"NeurIPS", r"NIPS", r"Neural Information Processing Systems"],
    "ICML": [r"ICML", r"International Conference on Machine Learning"],
    "ICLR": [r"ICLR", r"International Conference on Learning Representations"],
    "CVPR": [r"CVPR", r"Conference on Computer Vision and Pattern Recognition"],
    "ICCV": [r"ICCV", r"International Conference on Computer Vision"],
    "ECCV": [r"ECCV", r"European Conference on Computer Vision"],
    "ACL": [r"ACL"],
    "EMNLP": [r"EMNLP"],
    "NAACL": [r"NAACL"],
    "AAAI": [r"AAAI"],
    "IJCAI": [r"IJCAI"],
    "KDD": [r"KDD", r"SIGKDD"],
    "SIGIR": [r"SIGIR"],
    "TMLR": [r"TMLR", r"Transactions on Machine Learning Research"],
    "JMLR": [r"JMLR", r"Journal of Machine Learning Research"],
}

# Journals are often cited without a year next to the name; conferences need one
# ("ICLR" alone is too ambiguous: "we compare to ICLR baselines").
_YEAR_OPTIONAL = {"Nature", "Nature Machine Intelligence", "TMLR", "JMLR"}

# Separators people put between venue and year: "CVPR 2020", "CVPR'20", "ICML-2019",
# "(NeurIPS) 2020", "NeurIPS, 2021".
# Two-digit years only after an apostrophe ("CVPR'19"): a bare "Systems 23" or
# "Research, 13(Aug)" is a volume number, not a year.
_YEAR = r"(?:[\s\-(),]*((?:19|20)\d{2})\b|['\u2019](\d{2})\b)"  # \u2019 = typographic apostrophe

_PATTERNS = [
    (venue, re.compile(rf"\b(?:{'|'.join(alts)})\b(?:{_YEAR})?"))
    for venue, alts in _ALIASES.items()
]

_REJECT = re.compile(
    # "SDProc @ ACL 2024" is a workshop held at ACL; tiny papers are a non-archival track.
    r"submitted|under review|in submission|rejected|workshop|challenge|symposium|tutorial"
    r"|\s@\s?|tiny paper",
    re.IGNORECASE,
)
_FINDINGS = re.compile(r"\bfindings\b", re.IGNORECASE)


def _parse_year(four: str | None, two: str | None) -> int | None:
    if four is not None:
        year = int(four)
    elif two is not None:
        year = int(two) + (1900 if int(two) > 80 else 2000)  # NIPS'98 -> 1998
    else:
        return None
    return year if 1987 <= year <= 2030 else None


def _search(text: str) -> VenueMatch | None:
    """Earliest venue mention in `text` that passes the year rules."""
    best: tuple[int, VenueMatch] | None = None
    for venue, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            year = _parse_year(m.group(1), m.group(2))
            if year is None and venue not in _YEAR_OPTIONAL:
                continue
            name = (
                f"{venue} Findings"
                if venue in {"ACL", "EMNLP", "NAACL"} and _FINDINGS.search(text)
                else venue
            )
            if best is None or m.start() < best[0]:
                best = (m.start(), VenueMatch(name, year, text))
            break  # first valid match of this venue is enough
    return best[1] if best else None


def extract_venue(comments: str | None, journal_ref: str | None) -> VenueMatch | None:
    """The venue a paper was accepted at, or None if unknown or uncertain.

    `journal-ref` is checked first: it is set when the paper is formally published.
    """
    for field in (journal_ref, comments):
        if not field:
            continue
        text = " ".join(field.split())  # collapse arXiv's line breaks and indentation
        if _REJECT.search(text):
            continue
        match = _search(text)
        if match:
            return match
    return None
