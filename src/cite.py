"""APA 7 citations from data/papers.csv.

    from src.cite import load_papers, in_text, reference
    papers = load_papers()
    in_text(papers["li2020"], page=5)   # (Li et al., 2020, p. 5)
    reference(papers["li2020"])         # Li, J., Xiong, C., Socher, R., & Hoi, S. (2020). ...

papers.csv format: authors as "Surname, Given names" separated by ";", e.g.
    "Li, Junnan; Xiong, Caiming; Socher, Richard; Hoi, Steven"

Three kinds of source are recognised from the venue / link:
  - arXiv preprint      ... (2020). *Title* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2003.01285
  - conference paper    ... (2019). Title. In *Proceedings of ...*. https://...
  - journal article     ... (2022). Title. *IEEE Transactions on Image Processing*. https://doi.org/...

Notes:
  - APA writes article titles in sentence case ("Towards noise-resistant object
    detection with noisy annotations"). Titles are used exactly as written in
    papers.csv, so write them there in the case you want; automatic conversion
    would break names like R-CNN or PASCAL VOC.
  - Page numbers are PDF page numbers. For a journal or proceedings PDF whose
    printed page numbers differ (e.g. starting at 1234), adjust before citing.
  - Optional papers.csv columns volume, issue and pages are used when present
    (journals: "*Journal*, *31*(4), 120–135"; proceedings: "(pp. 1–10)").
  - Use reference(..., markdown=True) for *italics*; plain text has none.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_META = ROOT / "data" / "papers.csv"

ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.IGNORECASE)
ARXIV_VENUE_ID = re.compile(r"arXiv:(\d{4}\.\d{4,5})", re.IGNORECASE)
PROCEEDINGS = re.compile(r"\b(proceedings|conference|workshops?|symposium)\b", re.IGNORECASE)


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: list[tuple[str, str]]  # (surname, given names)
    year: str
    venue: str
    doi_or_url: str = ""
    volume: str = ""  # optional columns in papers.csv
    issue: str = ""
    pages: str = ""  # e.g. "1234-1245"

    def __post_init__(self):
        # APA page ranges use an en dash: 1234–1245.
        self.pages = re.sub(r"\s*-{1,2}\s*", "–", self.pages)

    @property
    def kind(self) -> str:
        if "arxiv" in self.venue.lower() or (not self.venue and "arxiv.org" in self.doi_or_url):
            return "preprint"
        if PROCEEDINGS.search(self.venue):
            return "proceedings"
        return "journal"


def parse_authors(text: str) -> list[tuple[str, str]]:
    authors = []
    for part in text.split(";"):
        part = " ".join(part.split())
        if not part:
            continue
        if "," in part:
            surname, given = (s.strip() for s in part.split(",", 1))
        else:  # "Junnan Li" -> surname is the last word
            *given_words, surname = part.split(" ")
            given = " ".join(given_words)
        authors.append((surname, given))
    return authors


def load_papers(meta_path: Path = DEFAULT_META) -> dict[str, Paper]:
    meta = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
    papers = {}
    for row in meta.itertuples(index=False):
        papers[row.paper_id] = Paper(
            paper_id=row.paper_id,
            title=row.title.strip(),
            authors=parse_authors(row.authors),
            year=row.year.strip() or "n.d.",
            venue=row.venue.strip(),
            doi_or_url=getattr(row, "doi_or_url", "").strip(),
            volume=getattr(row, "volume", "").strip(),
            issue=getattr(row, "issue", "").strip(),
            pages=getattr(row, "pages", "").strip(),
        )
    return papers


# ---------------------------------------------------------------- authors


def initials(given: str) -> str:
    """'Junnan' -> 'J.', 'Larry S.' -> 'L. S.', 'Bin-Bin' -> 'B.-B.'"""
    out = []
    for word in given.replace(".", ". ").split():
        pieces = [p for p in word.split("-") if p]
        if pieces:
            out.append("-".join(f"{p[0].upper()}." for p in pieces))
    return " ".join(out)


def format_author(surname: str, given: str) -> str:
    return f"{surname}, {initials(given)}" if given else surname


def author_list(authors: list[tuple[str, str]]) -> str:
    """APA 7 reference-list authors: up to 20 listed; 21+ = first 19, ..., last."""
    names = [format_author(s, g) for s, g in authors]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) <= 20:
        return ", ".join(names[:-1]) + ", & " + names[-1]
    return ", ".join(names[:19]) + ", . . . " + names[-1]


def in_text_authors(authors: list[tuple[str, str]]) -> str:
    """APA 7 in-text: 'Li', 'Xu & Bai', or 'Li et al.' for three or more."""
    surnames = [s for s, _ in authors]
    if not surnames:
        return "Anonymous"
    if len(surnames) == 1:
        return surnames[0]
    if len(surnames) == 2:
        return f"{surnames[0]} & {surnames[1]}"
    return f"{surnames[0]} et al."


# ---------------------------------------------------------------- citations


def in_text(paper: Paper, page: int | None = None) -> str:
    """Parenthetical citation, e.g. (Li et al., 2020, p. 5)."""
    parts = [in_text_authors(paper.authors), paper.year]
    if page is not None:
        parts.append(f"p. {page}")
    return f"({', '.join(parts)})"


def link(paper: Paper) -> str:
    """arXiv links become the arXiv DOI; other DOIs become https://doi.org/..."""
    m = ARXIV_ID.search(paper.doi_or_url) or ARXIV_VENUE_ID.search(paper.venue)
    if m and paper.kind == "preprint":
        return f"https://doi.org/10.48550/arXiv.{m.group(1)}"
    url = paper.doi_or_url
    if url.lower().startswith("10."):
        return f"https://doi.org/{url}"
    return url


def _end(text: str) -> str:
    """Add a full stop unless the text already ends with punctuation."""
    return text if text[-1:] in ".?!" else text + "."


def reference(paper: Paper, markdown: bool = False) -> str:
    """APA 7 reference-list entry."""
    it = (lambda s: f"*{s}*") if markdown else (lambda s: s)
    head = f"{_end(author_list(paper.authors))} ({paper.year})."
    title = paper.title
    if paper.kind == "preprint":
        # "[Preprint]" goes after the title, so the title's own punctuation moves after it.
        body = f"{it(title.rstrip('.'))} [Preprint]. arXiv."
    elif paper.kind == "proceedings":
        pages = f" (pp. {paper.pages})" if paper.pages else ""
        body = f"{_end(title)} In {it(paper.venue)}{pages}."
    elif paper.venue:
        # Journal: *Journal*, *volume*(issue), pages.
        source = it(paper.venue)
        if paper.volume:
            source += f", {it(paper.volume)}" + (f"({paper.issue})" if paper.issue else "")
        if paper.pages:
            source += f", {paper.pages}"
        body = f"{_end(title)} {source}."
    else:
        body = _end(title)
    url = link(paper)
    return " ".join(p for p in (head, body, url) if p)
