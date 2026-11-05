"""Tests for src/cite.py. Run: python -m pytest -q"""

from src.cite import Paper, author_list, in_text, initials, load_papers, parse_authors, reference


def paper(authors, kind="preprint", **kw):
    venues = {
        "preprint": "arXiv preprint arXiv:2003.01285",
        "proceedings": "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        "journal": "IEEE Transactions on Image Processing",
    }
    defaults = dict(paper_id="p", title="A title", year="2020", venue=venues[kind], doi_or_url="")
    defaults.update(kw)
    return Paper(authors=parse_authors(authors), **defaults)


def test_initials():
    assert initials("Junnan") == "J."
    assert initials("Larry S.") == "L. S."
    assert initials("Seung Hwan") == "S. H."
    assert initials("Bin-Bin") == "B.-B."


def test_parse_authors_both_formats():
    assert parse_authors("Li, Junnan; Hoi, Steven") == [("Li", "Junnan"), ("Hoi", "Steven")]
    assert parse_authors("Junnan Li; Steven C. H. Hoi") == [("Li", "Junnan"), ("Hoi", "Steven C. H.")]


def test_in_text_one_two_and_many_authors():
    assert in_text(paper("Li, Junnan"), 5) == "(Li, 2020, p. 5)"
    assert in_text(paper("Xu, Mengmeng; Bai, Yancheng"), 5) == "(Xu & Bai, 2020, p. 5)"
    assert in_text(paper("Xu, M; Bai, Y; Ghanem, B"), 5) == "(Xu et al., 2020, p. 5)"
    assert in_text(paper("Li, Junnan")) == "(Li, 2020)"


def test_reference_author_list_uses_ampersand():
    assert author_list(parse_authors("Li, Junnan; Xiong, Caiming; Hoi, Steven")) == "Li, J., Xiong, C., & Hoi, S."


def test_more_than_20_authors_are_shortened():
    names = "; ".join(f"Author{i}, Given" for i in range(1, 25))
    text = author_list(parse_authors(names))
    assert text.startswith("Author1, G., ") and ", . . . Author24, G." in text
    assert "Author19, G." in text and "Author20, G." not in text


def test_arxiv_preprint_reference():
    p = paper(
        "Li, Junnan; Hoi, Steven",
        title="Towards noise-resistant object detection with noisy annotations",
        doi_or_url="https://arxiv.org/abs/2003.01285",
    )
    assert reference(p) == (
        "Li, J., & Hoi, S. (2020). Towards noise-resistant object detection with noisy annotations "
        "[Preprint]. arXiv. https://doi.org/10.48550/arXiv.2003.01285"
    )
    assert "*Towards noise-resistant object detection with noisy annotations*" in reference(p, markdown=True)


def test_proceedings_reference_with_pages():
    p = paper("Xu, Mengmeng", kind="proceedings", title="Missing labels in object detection", pages="1-10")
    assert reference(p, markdown=True) == (
        "Xu, M. (2020). Missing labels in object detection. In *Proceedings of the IEEE/CVF Conference "
        "on Computer Vision and Pattern Recognition* (pp. 1–10)."
    )


def test_journal_reference_with_volume_issue_pages_and_doi():
    p = paper(
        "Wang, Shaoru; Gao, Jin", kind="journal", title="Narrowing the gap",
        doi_or_url="10.1109/TIP.2022.3211468", volume="31", issue="4", pages="6369--6380",
    )
    assert reference(p, markdown=True) == (
        "Wang, S., & Gao, J. (2020). Narrowing the gap. *IEEE Transactions on Image Processing*, "
        "*31*(4), 6369–6380. https://doi.org/10.1109/TIP.2022.3211468"
    )


def test_title_ending_in_question_mark_gets_no_extra_full_stop():
    p = paper("Li, Junnan", kind="journal", title="Do detectors overfit?")
    assert "Do detectors overfit? IEEE" in reference(p)


def test_load_papers_reads_the_project_csv():
    papers = load_papers()
    assert papers, "data/papers.csv is empty"
    for p in papers.values():
        assert p.authors and p.year and p.title, p.paper_id
        assert reference(p)  # every paper can be formatted
