"""Day 3: split a user's paragraph into individual claims.

Each sentence of the paragraph is one claim. For searching, citation markers
the writer already added ("[12]", "(Wu et al., 2018)") are removed, because
they carry no meaning for the embedding model and only add noise. The
original sentence is kept for display.

    from src.claims import split_claims
    for c in split_claims("Label noise hurts detectors [3]. Missing boxes matter less."):
        print(c.index, c.query)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import spacy

from src.ingest import ABBREVIATIONS

MIN_CLAIM_WORDS = 4  # "See Table 2." is not a claim

NUMERIC_CITATION = re.compile(r"\s*\[\d+(?:\s*[,–\-]\s*\d+)*\]")  # [3], [3, 7], [3-5]
AUTHOR_YEAR_CITATION = re.compile(
    r"\s*\((?:see\s+|e\.g\.,?\s+)?[A-Z][^()]*?\d{4}[a-z]?(?:,\s*p+\.\s*\d+)?\)"  # (Wu et al., 2018)
)
YEAR_ONLY_CITATION = re.compile(r"\s*\(\d{4}[a-z]?\)")  # "Wu et al. (2018)"
SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
WHITESPACE = re.compile(r"\s+")


@dataclass
class Claim:
    index: int  # 1-based position in the paragraph
    text: str  # the sentence as the user wrote it
    query: str  # cleaned text used for searching


_NLP: spacy.Language | None = None


def _nlp() -> spacy.Language:
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger", "attribute_ruler"])
    return _NLP


def strip_citations(text: str) -> str:
    text = NUMERIC_CITATION.sub("", text)
    text = AUTHOR_YEAR_CITATION.sub("", text)
    text = YEAR_ONLY_CITATION.sub("", text)
    text = SPACE_BEFORE_PUNCT.sub(r"\1", text)
    return WHITESPACE.sub(" ", text).strip()


def split_claims(paragraph: str, min_words: int = MIN_CLAIM_WORDS) -> list[Claim]:
    """Split a paragraph into claims, one per sentence."""
    paragraph = WHITESPACE.sub(" ", paragraph).strip()
    if not paragraph:
        return []

    sentences: list[str] = []
    for sent in _nlp()(paragraph).sents:
        text = sent.text.strip()
        if not text:
            continue
        # Glue back splits after "et al." / "e.g." etc., as in ingest, and
        # fragments that are only a citation, e.g. "[3, 7]." after "detection".
        only_citation = not any(ch.isalpha() for ch in strip_citations(text))
        if sentences and (only_citation or sentences[-1].lower().endswith(ABBREVIATIONS)):
            sentences[-1] = f"{sentences[-1]} {text}"
        else:
            sentences.append(text)

    claims = []
    for text in sentences:
        query = strip_citations(text)
        if len(query.split()) >= min_words:
            claims.append(Claim(len(claims) + 1, text, query))
    return claims
