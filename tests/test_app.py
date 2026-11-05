"""Tests for app.py, run headlessly with Streamlit's AppTest.

The real index, re-ranker and LLM are replaced with the same stand-ins the
other tests use, so this runs offline in a few seconds.

Run: python -m pytest -q
"""

from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import src.cite
import src.retrieve
from src.index import EvidenceIndex, build_index
from src.judge import Judge, LLMError
from tests.test_index import BagOfWords
from tests.test_pipeline import PAPERS, SENTENCES, KeywordLLM
from tests.test_retrieve import KeywordReranker

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PARAGRAPH = "Noisy bounding boxes reduce detector accuracy [2]. Espresso tastes better with oat milk."


@pytest.fixture()
def fakes(tmp_path, monkeypatch):
    table = pd.DataFrame(SENTENCES, columns=["paper_id", "page", "sentence"])
    table.insert(0, "sentence_id", range(len(table)))
    table.to_csv(tmp_path / "sentences.csv", index=False)
    index = build_index(tmp_path, model=BagOfWords(), progress=False)
    llm = KeywordLLM()

    monkeypatch.setattr(EvidenceIndex, "load", classmethod(lambda cls, *a, **k: index))
    monkeypatch.setattr(src.retrieve, "load_reranker", lambda *a, **k: KeywordReranker("reduce"))
    monkeypatch.setattr(Judge, "from_env", classmethod(lambda cls, *a, **k: Judge(llm, None, sleep=lambda s: None)))
    monkeypatch.setattr(src.cite, "load_papers", lambda *a, **k: PAPERS)
    st.cache_resource.clear()
    st.cache_data.clear()
    yield llm
    st.cache_resource.clear()
    st.cache_data.clear()


def start() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception, at.exception
    at.sidebar.slider[0].set_value(0.30).run()  # the stand-in encoder gives lower scores
    return at


def find_evidence(at: AppTest, paragraph: str = PARAGRAPH) -> AppTest:
    at.text_area[0].input(paragraph)
    at.button[0].click().run()  # "Find evidence"
    assert not at.exception, at.exception
    return at


def text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown) + "\n".join(c.value for c in at.caption)


def test_page_loads_and_lists_indexed_papers(fakes):
    at = start()
    assert at.title[0].value.endswith("CiteAssist")
    sidebar = "\n".join(m.value for m in at.sidebar.markdown)
    assert "Alpha paper" in sidebar and "Beta paper" in sidebar
    assert "Indexed papers (2)" in [h.value for h in at.sidebar.header]


def test_results_show_labels_quotes_citations_and_no_evidence(fakes):
    at = find_evidence(start())
    page = text(at)
    assert ":green-badge[Supports]" in page and ":red-badge[Contradicts]" in page
    assert "(Delta, 2021, p. 5)" in page
    assert "Noisy bounding boxes do not reduce detector accuracy" in page  # the quote itself
    assert "**Why:** rule supports" in page
    assert any("No sentence in the indexed papers" in i.value for i in at.info)  # espresso claim
    assert {m.label: m.value for m in at.metric}["Claims"] == "2"
    assert "References" in [s.value for s in at.subheader]


def test_example_button_fills_the_text_box(fakes):
    at = start()
    at.button[1].click().run()  # "Try an example"
    assert at.text_area[0].value.startswith("Detectors such as Faster R-CNN")


def test_empty_paragraph_asks_for_text(fakes):
    at = start()
    at.button[0].click().run()
    assert any("Paste a paragraph first" in w.value for w in at.warning)


def test_judge_switched_off_shows_evidence_without_labels(fakes):
    at = start()
    at.sidebar.toggle[0].set_value(False).run()
    at = find_evidence(at)
    assert ":blue-badge[Evidence]" in text(at)
    assert fakes.calls == 0


def test_missing_llm_key_warns_but_still_works(fakes, monkeypatch):
    def no_key(cls, *a, **k):
        raise SystemExit("GEMINI_API_KEY is not set.")

    monkeypatch.setattr(Judge, "from_env", classmethod(no_key))
    at = find_evidence(start())
    assert any("GEMINI_API_KEY is not set" in w.value for w in at.sidebar.warning)
    assert ":blue-badge[Evidence]" in text(at)


def test_llm_failure_falls_back_to_unlabelled_evidence(fakes, monkeypatch):
    def broken(self, system, user):
        raise LLMError("gave up after 4 attempts: HTTP 429")

    monkeypatch.setattr(KeywordLLM, "complete", broken)
    at = find_evidence(start())
    assert any("could not be reached" in w.value for w in at.warning)
    assert ":blue-badge[Evidence]" in text(at)


def test_missing_index_shows_a_clear_error(fakes, monkeypatch):
    def missing(cls, *a, **k):
        raise SystemExit("No index in index/. Run `python -m src.index build` first.")

    monkeypatch.setattr(EvidenceIndex, "load", classmethod(missing))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert any("src.index build" in e.value for e in at.error)
