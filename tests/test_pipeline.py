"""End-to-end test of src/pipeline.py with stand-in models (offline).

Run: python -m pytest -q
"""

import json

import pandas as pd
import pytest

from src.cite import Paper, parse_authors
from src.index import build_index
from src.judge import Judge
from src.pipeline import CiteAssist
from src.retrieve import Retriever
from tests.test_index import BagOfWords

SENTENCES = [
    ("alpha2020", 4, "Noisy bounding boxes reduce detector accuracy sharply at high noise rates."),
    ("alpha2020", 2, "We study noisy bounding boxes and detector accuracy in detail."),
    ("beta2021", 5, "Noisy bounding boxes do not reduce detector accuracy in our experiments."),
]
PAPERS = {
    "alpha2020": Paper("alpha2020", "Alpha paper", parse_authors("Alpha, Ann; Beta, Bo; Gamma, Cy"), "2020", "arXiv preprint"),
    "beta2021": Paper("beta2021", "Beta paper", parse_authors("Delta, Di"), "2021", "Some Journal"),
}


class KeywordLLM:
    """Fake LLM: 'not' -> contradicts, 'reduce' -> supports, else related."""

    name, model = "fake", "m"

    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        items = []
        for i, line in enumerate((l for l in user.splitlines() if l.startswith("<evidence")), 1):
            label = "contradicts" if " not " in line else "supports" if "reduce" in line else "related"
            items.append({"id": i, "label": label, "reason": f"rule {label}"})
        return json.dumps({"judgements": items})


@pytest.fixture()
def assistant(tmp_path):
    table = pd.DataFrame(SENTENCES, columns=["paper_id", "page", "sentence"])
    table.insert(0, "sentence_id", range(len(table)))
    table.to_csv(tmp_path / "sentences.csv", index=False)
    index = build_index(tmp_path, model=BagOfWords(), progress=False)
    judge = Judge(KeywordLLM(), cache_path=None, sleep=lambda s: None)
    return CiteAssist(Retriever(index, threshold=0.3), judge, PAPERS)


def test_full_pipeline_labels_cites_and_orders(assistant):
    paragraph = "Noisy bounding boxes reduce detector accuracy [2]. Espresso tastes better with oat milk."
    checked = assistant.check(paragraph)
    assert [c.claim.index for c in checked] == [1, 2]

    first = checked[0]
    labels = [i.label for i in first.items]
    assert labels == sorted(labels, key=["supports", "contradicts", "related"].index)  # supports first
    assert first.verdict == "contested"  # one supports, one contradicts
    cites = {i.paper.paper_id: i.citation for i in first.items}
    assert cites["alpha2020"] == "(Alpha et al., 2020, p. 4)" or cites["alpha2020"] == "(Alpha et al., 2020, p. 2)"
    assert cites["beta2021"] == "(Delta, 2021, p. 5)"

    assert not checked[1].found and checked[1].verdict == "no evidence"


def test_one_llm_call_per_claim_and_none_without_evidence(assistant):
    assistant.check("Noisy bounding boxes reduce detector accuracy. Espresso tastes better with oat milk.")
    assert assistant.judge.provider.calls == 1  # the espresso claim never reaches the LLM


def test_pipeline_without_judge_has_no_labels(assistant):
    assistant.judge = None
    checked = assistant.check("Noisy bounding boxes reduce detector accuracy.")
    assert checked[0].found and all(i.label is None for i in checked[0].items)
    assert checked[0].verdict == "found"
