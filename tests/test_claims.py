"""Tests for src/claims.py. Run: python -m pytest -q"""

from src.claims import split_claims, strip_citations

PARAGRAPH = (
    "Label noise is a major problem in object detection [3, 7]. "
    "As shown by Wu et al. (2018), Faster R-CNN is fairly robust to missing annotations. "
    "However, noisy bounding boxes cause a large drop in mAP (Zhou et al., 2023). "
    "See Table 2. "
    "Do detectors overfit to noisy labels (e.g., Li et al., 2020)?"
)


def test_one_claim_per_sentence_and_short_ones_dropped():
    claims = split_claims(PARAGRAPH)
    assert [c.index for c in claims] == [1, 2, 3, 4]  # "See Table 2." is dropped
    assert claims[0].text == "Label noise is a major problem in object detection [3, 7]."


def test_et_al_does_not_split_a_claim():
    assert split_claims(PARAGRAPH)[1].text.startswith("As shown by Wu et al. (2018), Faster R-CNN")


def test_citations_removed_from_query_but_kept_in_text():
    claims = split_claims(PARAGRAPH)
    assert claims[0].query == "Label noise is a major problem in object detection."
    assert claims[2].query == "However, noisy bounding boxes cause a large drop in mAP."
    assert claims[3].query == "Do detectors overfit to noisy labels?"
    assert "(Zhou et al., 2023)" in claims[2].text


def test_strip_citations_variants():
    assert strip_citations("Noise hurts [12].") == "Noise hurts."
    assert strip_citations("Noise hurts [3-5].") == "Noise hurts."
    assert strip_citations("Noise hurts (Xu, Bai, & Ghanem, 2019).") == "Noise hurts."
    assert strip_citations("Noise hurts (Li et al., 2020, p. 4).") == "Noise hurts."
    assert strip_citations("Wu et al. (2018) showed this.") == "Wu et al. showed this."
    # Brackets that are not citations stay.
    assert strip_citations("Accuracy (mAP) drops by 4 points.") == "Accuracy (mAP) drops by 4 points."


def test_empty_paragraph():
    assert split_claims("   ") == []