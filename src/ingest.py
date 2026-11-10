"""Read research-paper PDFs and turn them into a clean sentence table.

Pipeline for each PDF listed in data/papers.csv:
  1. Extract text blocks page by page with PyMuPDF.
  2. Drop running headers/footers, page numbers, arXiv stamps and the
     reference list (everything from the "References" heading onwards).
  3. Clean the text: fix hyphenated line breaks, join broken lines.
  4. Split the whole paper into sentences with spaCy, keeping the page
     each sentence starts on (so sentences crossing a page break survive).
  5. Drop sentences that are too short, too long, or mostly symbols.

Output: index/sentences.csv with columns
  sentence_id, paper_id, page, sentence

Run from the project root:
    python -m src.ingest
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf  # PyMuPDF (the old name `fitz` is deprecated in 2.0)
import pandas as pd
import spacy

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAPERS_DIR = ROOT / "data" / "papers"
DEFAULT_META = ROOT / "data" / "papers.csv"
DEFAULT_OUT = ROOT / "index" / "sentences.csv"

REQUIRED_COLUMNS = ["paper_id", "file", "title", "authors", "year", "venue"]

# A block whose whole text is one of these headings starts the reference list.
REFERENCES_HEADING = re.compile(
    r"^\s*(?:\d+\.?\s*|[IVX]+\.\s*)?(references|bibliography|literature cited)\s*$",
    re.IGNORECASE,
)
# Things that are never useful evidence.
ARXIV_STAMP = re.compile(r"arXiv:\d{4}\.\d{4,5}(v\d+)?", re.IGNORECASE)
PAGE_NUMBER = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.IGNORECASE)
URL_OR_EMAIL = re.compile(r"(https?://|www\.|\S+@\S+\.\w+)")
# The acknowledgements section is skipped up to the next section heading.
ACK_HEADING = re.compile(r"^\s*(?:\d+\.?\s*)?acknowledge?ments?\.?\s*$", re.IGNORECASE)
# Backup for acknowledgements without their own heading line.
THANKS = re.compile(
    r"\b(we (would like to |wish to )?(thank|acknowledge)|grateful|gratitude"
    r"|(this|the) (work|research|project) (was|is) (partially |partly |in part )?(supported|funded)"
    r"|grant (no\.?|number))",
    re.IGNORECASE,
)
# Words allowed in a list of names ("A, B, and C"); everything else lowercase counts as prose.
NAME_LIST_WORDS = {"and", "&"}
MIN_LOWERCASE_FRACTION = 0.25
HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")  # "detec-\ntion" -> "detection"
WHITESPACE = re.compile(r"\s+")
# A section heading on its own line: optional number, a few capitalised words,
# no full stop at the end. e.g. "1. Introduction", "3.2 Box jitter", "Abstract".
HEADING = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVX]+\.)?\s*[A-Z][\w\-:&,' ]*$")
MAX_HEADING_WORDS = 6
# A common section name printed on the same line as the text that follows it,
# e.g. "Conclusion We evaluated ..." or "4 Discussion Our research ...".
# Only stripped when the next word is capitalised, so "Results show that ..."
# is left alone.
SECTION_NAMES = (
    r"(?:Abstract|Introduction|Related Work|Background|Preliminaries|Method|Methods|Methodology"
    r"|Approach|Experiments?|Experimental Setup|Results|Discussion|Conclusions?"
    r"|Limitations|Future Work)"
)
INLINE_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+|[IVX]+\.\s+)?" + SECTION_NAMES + r"\s*[:.\-—]?\s+(?=[A-Z])"
)
# A numbered sub-heading stuck to its text, e.g. "Experiments 4.1 Dataset We manually ..."
# or "3.2 Implementation Details We train ...": the number must have a dot (4.1),
# and the heading is 1-4 capitalised words before a word that typically starts a sentence.
SENTENCE_STARTERS = (
    r"(?:We|Our|In|The|This|These|To|For|As|A|An|It|Here|First|Next|Finally|Following"
    r"|Unlike|Since|Given|Each|All|Table|Figure|Fig)\b"
)
SUBHEADING = re.compile(
    r"^(?:" + SECTION_NAMES + r"\s+)?\d+(?:\.\d+)+\.?\s+(?:[A-Z][\w\-]*\s+){1,4}?(?=" + SENTENCE_STARTERS + ")"
)
# Figure and table captions: "Figure 1: The mean ...", "Table 2. Results on ...".
CAPTION = re.compile(r"^(?:Figure|Fig\.?|Table|Tab\.)\s*\d+[a-z]?\s*[:.|]", re.IGNORECASE)
CLOSING = "\"'”’)]"
# Abbreviations that spaCy may wrongly treat as the end of a sentence.
ABBREVIATIONS = ("fig.", "figs.", "eq.", "eqs.", "sec.", "tab.", "al.", "e.g.", "i.e.", "vs.", "cf.", "no.")
# Marker that forces a sentence boundary where a heading used to be.
# It never reaches spaCy: the text is split on it first.
SEGMENT_BREAK = "\x00"

# Fraction of the page height treated as the header / footer zone.
MARGIN_FRACTION = 0.07


@dataclass
class PaperResult:
    paper_id: str
    pages: int
    sentences: list[tuple[int, str]]  # (page, sentence)
    references_page: int | None
    dropped: int


# ---------------------------------------------------------------- extraction


def _normalise_for_repeat(text: str) -> str:
    """Key used to spot running headers: lowercase, digits replaced."""
    return re.sub(r"\d+", "#", WHITESPACE.sub(" ", text).strip().lower())


def _page_blocks(page: pymupdf.Page) -> list[tuple[float, float, str]]:
    """Return (y0, y1, text) for each text block, in PDF reading order.

    PyMuPDF's default order follows the content stream, which for most
    LaTeX two-column papers is already column-by-column. We deliberately
    do NOT use sort=True, because that interleaves the two columns.
    """
    blocks = []
    for x0, y0, x1, y1, text, _no, block_type in page.get_text("blocks"):
        if block_type != 0:  # 1 = image block
            continue
        text = text.strip()
        if text:
            blocks.append((y0, y1, text))
    return blocks


def extract_pages(pdf_path: Path) -> tuple[list[str], int | None]:
    """Extract cleaned text for each page, stopping at the reference list.

    Returns (page_texts, references_page). page_texts[i] is page i+1.
    """
    doc = pymupdf.open(pdf_path)
    try:
        raw = [(page.rect.height, _page_blocks(page)) for page in doc]
    finally:
        doc.close()

    # 1) Find running headers/footers: short blocks in the top/bottom margin
    #    that repeat (ignoring numbers) on at least 30% of pages.
    margin_counts: Counter[str] = Counter()
    for height, blocks in raw:
        seen = set()
        for y0, y1, text in blocks:
            in_margin = y1 < height * MARGIN_FRACTION or y0 > height * (1 - MARGIN_FRACTION)
            if in_margin and len(text) < 150:
                seen.add(_normalise_for_repeat(text))
        margin_counts.update(seen)
    min_repeats = max(2, int(0.3 * len(raw)))
    running = {key for key, n in margin_counts.items() if n >= min_repeats}

    # 2) Walk the pages, dropping junk blocks, until the References heading.
    page_texts: list[str] = []
    references_page = None
    min_ref_page = max(1, len(raw) // 3)  # ignore a "References" mention in the intro
    in_acknowledgements = False  # carries across blocks and pages
    for page_no, (height, blocks) in enumerate(raw, start=1):
        kept = []
        for y0, y1, text in blocks:
            if _normalise_for_repeat(text) in running:
                continue
            if PAGE_NUMBER.match(text) or ARXIV_STAMP.search(text):
                continue
            lines_out: list[str] = []
            prev = ""
            lines = text.splitlines()
            for i, line in enumerate(lines):
                stripped = line.strip()
                if page_no > min_ref_page and REFERENCES_HEADING.match(stripped):
                    references_page = page_no
                    break
                if ACK_HEADING.match(stripped):
                    in_acknowledgements = True  # skip until the next section heading
                    lines_out.append(SEGMENT_BREAK)
                    prev = ""
                    continue
                next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
                if is_heading(stripped, prev, next_line):
                    in_acknowledgements = False
                    lines_out.append(SEGMENT_BREAK)  # drop it, but end the sentence here
                    prev = ""  # a heading is a boundary, like the start of a block
                elif in_acknowledgements:
                    continue
                else:
                    lines_out.append(line)
                    prev = stripped
            kept.append("\n".join(lines_out))
            if references_page is not None:
                break
        page_texts.append(clean_text("\n".join(kept)))
        if references_page is not None:
            break
    return page_texts, references_page


def is_heading(line: str, prev_line: str, next_line: str) -> bool:
    """True for a short title-like line that starts a new section.

    Guards against dropping prose: the line must follow the start of a block
    or a line that ends a sentence, and the next line must not carry on the
    same sentence in lowercase ("...expensive to" / "produce, and ...").
    """
    if not line or len(line.split()) > MAX_HEADING_WORDS or line.endswith("."):
        return False
    if prev_line and prev_line[-1] not in ".:?!":
        return False
    if next_line[:1].islower():
        return False
    return bool(HEADING.match(line))


def clean_text(text: str) -> str:
    """Fix hyphenated line breaks and join wrapped lines into prose."""
    text = text.replace("­", "")  # soft hyphens
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")  # ligatures
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    text = WHITESPACE.sub(" ", text)
    return text.strip()


# ------------------------------------------------------------------ sentences


def is_good_sentence(sentence: str, min_words: int, max_words: int) -> bool:
    """Heuristic filter for sentences worth indexing as evidence."""
    words = sentence.split()
    if not (min_words <= len(words) <= max_words):
        return False
    letters = sum(ch.isalpha() for ch in sentence)
    if letters / max(len(sentence), 1) < 0.6:  # tables, equations, numbers
        return False
    if URL_OR_EMAIL.search(sentence):  # affiliations, footnote links
        return False
    if THANKS.search(sentence):  # acknowledgements and funding notes
        return False
    # Author lists and titles are almost all capitalised words and have no
    # full stop. Real sentences can be name-heavy ("We evaluate Faster R-CNN,
    # RetinaNet and FCOS on PASCAL VOC.") but still end like a sentence.
    # (Lists and titles without a full stop are also caught by the ending check below.)
    lowercase = sum(w[:1].islower() and w not in NAME_LIST_WORDS for w in words)
    if lowercase / len(words) < MIN_LOWERCASE_FRACTION / 2.5:
        return False
    # Fragments cut off by the page layout ("... such as Faster-RCNN",
    # "... formulated as μi =") don't end like a sentence.
    if sentence.rstrip().rstrip(CLOSING)[-1:] not in ".?!":
        return False
    if CAPTION.match(sentence):  # figure and table captions
        return False
    if not sentence[0].isalnum() and sentence[0] not in "\"'(“":  # fragments
        return False
    return True


def split_sentences(
    nlp: spacy.Language,
    page_texts: list[str],
    min_words: int,
    max_words: int,
) -> tuple[list[tuple[int, str]], int]:
    """Split a whole paper into sentences, tagging each with its start page."""
    # Join pages into one string so sentences crossing a page break stay whole,
    # and remember where each page starts.
    starts, parts, offset = [], [], 0
    for text in page_texts:
        starts.append(offset)
        parts.append(text)
        offset += len(text) + 1
    full = " ".join(parts)
    nlp.max_length = max(nlp.max_length, len(full) + 1)

    def page_of(char_index: int) -> int:
        page = 1
        for i, start in enumerate(starts, start=1):
            if char_index >= start:
                page = i
        return page

    # Cut the text at headings so a heading never merges into a sentence.
    segments, offset = [], 0
    for piece in full.split(SEGMENT_BREAK):
        if piece.strip():
            segments.append((offset, piece))
        offset += len(piece) + len(SEGMENT_BREAK)

    # (start_char, text) for every raw sentence, in order.
    raw: list[tuple[int, str]] = []
    for (seg_start, _), doc in zip(segments, nlp.pipe(seg for _, seg in segments)):
        for sent in doc.sents:
            text = sent.text.strip()
            if not text:
                continue
            # spaCy sometimes ends a sentence at "Fig." or "et al."; glue it back.
            if raw and raw[-1][1].lower().endswith(ABBREVIATIONS):
                raw[-1] = (raw[-1][0], f"{raw[-1][1]} {text}")
            else:
                raw.append((seg_start + sent.start_char, text))

    kept, dropped, seen = [], 0, set()
    for start, text in raw:
        text = SUBHEADING.sub("", text, count=1)
        text = INLINE_HEADING.sub("", text, count=1)
        if not is_good_sentence(text, min_words, max_words) or text in seen:
            dropped += 1
            continue
        seen.add(text)
        kept.append((page_of(start), text))
    return kept, dropped


# ----------------------------------------------------------------------- main


def load_metadata(meta_path: Path) -> pd.DataFrame:
    meta = pd.read_csv(meta_path, dtype=str).fillna("")
    missing = [c for c in REQUIRED_COLUMNS if c not in meta.columns]
    if missing:
        sys.exit(f"{meta_path} is missing columns: {', '.join(missing)}")
    dupes = meta["paper_id"][meta["paper_id"].duplicated()].tolist()
    if dupes:
        sys.exit(f"Duplicate paper_id values in {meta_path}: {', '.join(dupes)}")
    return meta


def ingest(
    papers_dir: Path = DEFAULT_PAPERS_DIR,
    meta_path: Path = DEFAULT_META,
    out_path: Path = DEFAULT_OUT,
    min_words: int = 8,
    max_words: int = 80,
) -> pd.DataFrame:
    meta = load_metadata(meta_path)
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger", "attribute_ruler"])

    results: list[PaperResult] = []
    for row in meta.itertuples(index=False):
        pdf_path = papers_dir / row.file
        if not pdf_path.exists():
            print(f"  ! {row.paper_id}: {row.file} not found in {papers_dir}, skipping")
            continue
        page_texts, ref_page = extract_pages(pdf_path)
        sentences, dropped = split_sentences(nlp, page_texts, min_words, max_words)
        results.append(PaperResult(row.paper_id, len(page_texts), sentences, ref_page, dropped))

    rows = [
        {"paper_id": r.paper_id, "page": page, "sentence": text}
        for r in results
        for page, text in r.sentences
    ]
    table = pd.DataFrame(rows, columns=["paper_id", "page", "sentence"])
    table.insert(0, "sentence_id", range(len(table)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    # Summary
    print(f"\n{'paper':<14}{'pages':>6}{'kept':>7}{'dropped':>9}  references from")
    for r in results:
        ref = f"p. {r.references_page}" if r.references_page else "not found ⚠"
        print(f"{r.paper_id:<14}{r.pages:>6}{len(r.sentences):>7}{r.dropped:>9}  {ref}")
    print(f"\n{len(table)} sentences from {len(results)} papers -> {out_path}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-words", type=int, default=8)
    parser.add_argument("--max-words", type=int, default=80)
    args = parser.parse_args()
    ingest(args.papers_dir, args.meta, args.out, args.min_words, args.max_words)


if __name__ == "__main__":
    main()
