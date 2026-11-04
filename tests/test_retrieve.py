"""Tests for src/retrieve.py.

Like test_index.py, these use a bag-of-words encoder instead of MiniLM, and a
keyword-based stand-in for the cross-encoder, so they run offline.

Run: python -m pytest -q
"""

import numpy as np
import pandas as pd
import pytest

from src.index import build_index
from src.retrieve import Retriever
from tests.test_index import BagOfWords

CLAIM = "Noisy bounding boxes reduce detector accuracy."

SENTENCES = [
    ("alpha2020", 1, "Noisy bounding boxes are common in large detection datasets."),
    ("alpha2020", 2, "We study noisy bounding boxes and detector accuracy in detail."),
    ("alpha2020", 3, "Noisy bounding boxes and detector accuracy are linked in many benchmarks."),
    ("alpha2020", 4, "Noisy bounding boxes reduce detector accuracy sharply at high noise rates."),
    ("beta2021", 5, "Detector accuracy drops when bounding boxes are noisy."),
    ("gamma2022", 6, "Coffee improves sleep quality for most adults."),
]


class KeywordReranker:
    """Stand-in cross-encoder: high logit if the sentence contains the keyword."""

    def __init__(self, keyword: str):
        self.keyword = keyword
        self.pairs_seen = []

    def predict(self, pairs, **kwargs):
        self.pairs_seen.extend(pairs)
        return np.array([4.0 if self.keyword in s.lower() else -2.0 for _, s in pairs])


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("retrieve")
    table = pd.DataFrame(SENTENCES, columns=["paper_id", "page", "sentence"])
    table.insert(0, "sentence_id", range(len(table)))
    table.to_csv(tmp / "sentences.csv", index=False)
    return build_index(tmp, model=BagOfWords(), progress=False)


def test_unrelated_claim_gets_no_evidence(index):
    result = Retriever(index).retrieve("Espresso tastes better with oat milk at breakfast.")
    assert not result.found
    assert result.evidence == []


def test_returns_at_most_top_k_ranked_from_one(index):
    result = Retriever(index, top_k=3, max_per_paper=None).retrieve(CLAIM)
    assert len(result.evidence) == 3
    assert [e.rank for e in result.evidence] == [1, 2, 3]
    assert all(e.similarity >= 0.40 for e in result.evidence)


def test_per_paper_cap_spreads_evidence(index):
    capped = Retriever(index, top_k=3, max_per_paper=2).retrieve(CLAIM)
    papers = [e.paper_id for e in capped.evidence]
    assert papers.count("alpha2020") <= 2
    assert "beta2021" in papers

    uncapped = Retriever(index, top_k=3, max_per_paper=None, threshold=0.0).retrieve(CLAIM)
    assert len(uncapped.evidence) == 3


def test_threshold_filters_weak_matches(index):
    strict = Retriever(index, threshold=0.99).retrieve(CLAIM)
    assert not strict.found
    loose = Retriever(index, threshold=0.0, top_k=10, max_per_paper=None).retrieve(CLAIM)
    assert len(loose.evidence) >= 5


def test_reranker_changes_order_and_scores_are_probabilities(index):
    reranker = KeywordReranker("drops")
    result = Retriever(index, reranker=reranker, max_per_paper=None).retrieve(CLAIM)
    top = result.evidence[0]
    assert top.paper_id == "beta2021"  # the only sentence with "drops"
    assert all(0.0 < e.rerank_score < 1.0 for e in result.evidence)
    scores = [e.rerank_score for e in result.evidence]
    assert scores == sorted(scores, reverse=True)


def test_threshold_is_applied_before_reranking(index):
    # "coffee" would win the re-ranking, but it is far below the threshold,
    # so the cross-encoder never even sees it.
    reranker = KeywordReranker("coffee")
    result = Retriever(index, reranker=reranker).retrieve(CLAIM)
    assert all("Coffee" not in e.sentence for e in result.evidence)
    assert all("Coffee" not in s for _, s in reranker.pairs_seen)


def test_without_reranker_scores_are_none(index):
    result = Retriever(index).retrieve(CLAIM)
    assert all(e.rerank_score is None for e in result.evidence)


def test_paragraph_gives_one_result_per_claim(index):
    paragraph = f"{CLAIM} Espresso tastes better with oat milk at breakfast [4]."
    results = Retriever(index).retrieve_paragraph(paragraph)
    assert [r.claim.index for r in results] == [1, 2]
    assert results[0].found and not results[1].found
