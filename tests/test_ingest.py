"""Tests for src/ingest.py on a deliberately messy synthetic paper.

The PDF has a running header, page numbers, an arXiv side stamp, an email,
a numeric table, hyphenated line breaks, a sentence that crosses a page break,
section headings on their own lines, "et al." / "Fig. 2", and a reference
list that must not be indexed.

Run: python -m pytest -q
"""

import textwrap

import pandas as pd
import pymupdf
import pytest

from src.ingest import ingest


def _para(text: str, width: int = 48) -> str:
    return "\n".join(textwrap.wrap(text, width))


def build_messy_pdf(path):
    doc = pymupdf.open()

    def page(left, right, num, stamp=False):
        p = doc.new_page(width=612, height=792)
        p.insert_text((150, 30), "Workshop on Learning with Noisy Labels, 2026", fontsize=8)
        p.insert_textbox(pymupdf.Rect(50, 70, 300, 740), left, fontsize=9)
        p.insert_textbox(pymupdf.Rect(312, 70, 562, 740), right, fontsize=9)
        p.insert_text((300, 775), str(num), fontsize=8)
        if stamp:
            p.insert_text((20, 600), "arXiv:2101.00001v2 [cs.CV] 4 Jan 2026", fontsize=10, rotate=90)

    page(
        "\n".join([
            "Label Noise in Object Detection", "",
            "Jane Doe*1, John Roe*1, Mary Major2, Richard Miles2, Alex Smith1, and Kim Lee2",
            "University of Somewhere",
            "jane.doe@example.edu", "", "Abstract",
            "Detectors trained with noisy bounding boxes showed a substantial drop in mAP compared with detec-",
            "tors trained on clean annotations. We study how missing annotations, class flips and box jitter",
            "affect modern two-stage detectors such as",
            "Faster R-CNN and Cascade",
            "R-CNN on PASCAL VOC and COCO.",
        ]),
        "\n".join([
            "1. Introduction",
            _para("Large annotated datasets are expensive to produce, and annotation errors are common in practice. "
                  "Wu et al. showed that missing annotations have a milder effect on detector accuracy than incorrect class labels. "
                  "In this paper we quantify the effect of each noise type separately. "
                  "This last sentence of the page continues onto the next page because papers often break sentences across"),
        ]),
        1,
        stamp=True,
    )
    page(
        "\n".join([
            _para("page boundaries and the ingest step should keep it whole."),
            "2. Method",
            _para("We corrupt the training annotations at rates from 10% to 50% and retrain Faster R-CNN each time."),
        ]),
        "\n".join([
            "Table 1. mAP under noise.", "10% 20% 30% 40%", "71.2 68.4 63.0 55.1", "",
            "3. Results",
            _para("As shown in Fig. 2, accuracy fell sharply once more than thirty percent of the boxes were jittered."),
        ]),
        2,
    )
    page(
        "\n".join([
            "4. Conclusion",
            _para("Localization noise is far more harmful than missing labels for two-stage detectors in our experiments."),
            "Acknowledgements",
            _para("We would like to express our gratitude to the anonymous reviewers for their many helpful comments. "
                  "Our lab members also gave useful feedback on early drafts of this paper."),
            "5. Limitations",
            _para("Our study only covers two-stage detectors evaluated on two public benchmark datasets."),
            "", "References",
            _para("[1] A. Author and B. Author. A paper about label noise in object detection that should not be indexed. In CVPR, 2019."),
        ]),
        _para("[2] C. Author. Another reference entry with enough words to pass the length filter easily. arXiv, 2020."),
        3,
    )
    doc.save(path)


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ingest")
    (tmp / "papers").mkdir()
    build_messy_pdf(tmp / "papers" / "test2026.pdf")
    (tmp / "papers.csv").write_text(
        "paper_id,file,title,authors,year,venue,doi_or_url\n"
        'test2026,test2026.pdf,Label Noise in Object Detection,"Doe, Jane; Roe, John",2026,Test,\n'
    )
    return ingest(tmp / "papers", tmp / "papers.csv", tmp / "sentences.csv")


def _all_text(table: pd.DataFrame) -> str:
    return " ".join(table["sentence"])


def test_columns(table):
    assert list(table.columns) == ["sentence_id", "paper_id", "page", "sentence"]


def test_hyphenated_word_is_joined(table):
    assert "detectors trained on clean annotations" in _all_text(table)
    assert "detec-" not in _all_text(table)


def test_short_capitalised_line_mid_sentence_is_kept(table):
    assert "such as Faster R-CNN and Cascade R-CNN on PASCAL VOC" in _all_text(table)


def test_sentence_crossing_page_break_is_whole_and_on_start_page(table):
    row = table[table["sentence"].str.startswith("This last sentence")].iloc[0]
    assert row["sentence"].endswith("should keep it whole.")
    assert row["page"] == 1


def test_headings_do_not_merge_into_sentences(table):
    for heading in ("Abstract", "Introduction", "Method", "Results", "Conclusion"):
        assert not table["sentence"].str.contains(heading).any(), heading
    assert table["sentence"].str.startswith("Large annotated datasets").any()


def test_abbreviations_do_not_split_sentences(table):
    assert table["sentence"].str.startswith("Wu et al. showed").any()
    assert table["sentence"].str.startswith("As shown in Fig. 2, accuracy").any()


def test_junk_is_removed(table):
    text = _all_text(table)
    for junk in ("Workshop on Learning", "arXiv:2101", "example.edu", "71.2"):
        assert junk not in text, junk


def test_author_list_is_removed(table):
    assert "Richard Miles" not in _all_text(table)


def test_acknowledgements_are_removed_but_next_section_is_kept(table):
    text = _all_text(table)
    assert "gratitude" not in text
    assert "early drafts" not in text  # no "thank" words, removed only by the section skip
    assert "Our study only covers two-stage detectors" in text


def test_references_are_cut(table):
    assert "should not be indexed" not in _all_text(table)
    assert "Another reference entry" not in _all_text(table)
    assert table["page"].max() == 3  # the conclusion before "References" is kept
