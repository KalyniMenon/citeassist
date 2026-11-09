"""Tests for evaluate.py: metrics are checked on small cases with known answers.

Run: python -m pytest -q
"""

import argparse
import json

import pandas as pd
import pytest

import evaluate as ev
from src.index import EvidenceIndex, build_index
from src.judge import Judge, LLMError
from tests.test_index import BagOfWords
from tests.test_retrieve import SENTENCES, KeywordReranker

GOLD = "Noisy bounding boxes reduce detector accuracy sharply at high noise rates."  # alpha2020 p. 4
CSV_HEADER = "claim,gold_paper,gold_page,gold_sentence,gold_label,notes\n"


@pytest.fixture()
def index(tmp_path):
    table = pd.DataFrame(SENTENCES, columns=["paper_id", "page", "sentence"])
    table.insert(0, "sentence_id", range(len(table)))
    table.to_csv(tmp_path / "sentences.csv", index=False)
    return build_index(tmp_path, model=BagOfWords(), progress=False)


def write_csv(path, rows):
    path.write_text(CSV_HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- gold matching + validation


def test_gold_sentence_matching(index):
    t = index.table
    assert ev.resolve_gold(t, "alpha2020", GOLD) == (3, "")
    # Quotes, spacing, case and a missing full stop don't matter.
    assert ev.resolve_gold(t, "alpha2020", '  "noisy bounding boxes   reduce detector accuracy sharply at high noise rates" ')[0] == 3
    sid, note = ev.resolve_gold(t, "alpha2020", GOLD.replace("sharply", "sharp"))
    assert sid == 3 and "approximately" in note
    assert ev.resolve_gold(t, "alpha2020", "A sentence that is not in any paper at all.")[0] is None


def test_load_test_set_reports_errors_and_warnings(index, tmp_path):
    path = write_csv(tmp_path / "t.csv", [
        f'Box noise hurts detectors a lot.,alpha2020,4,"{GOLD}",supports,',
        f'Box noise hurts detectors badly.,alpha2020,9,"{GOLD}",Supports,',  # wrong page, capital label
        f'"{GOLD}",alpha2020,4,"{GOLD}",supports,',  # claim copied from the sentence
        "Espresso is best with oat milk.,,,,none,",
        ",,,,,",  # blank template row: skipped
        "A claim with a bad label.,,,,maybe,",
        "A claim with no sentence.,alpha2020,4,,supports,",
        'A claim with a made-up sentence.,alpha2020,4,"This sentence does not exist anywhere.",supports,',
    ])
    claims, errors, warnings = ev.load_test_set(path, index.table)
    assert [c.row for c in claims] == [2, 3, 4, 5]  # Excel rows
    assert claims[1].gold_label == "supports" and claims[1].gold_page == 4  # page taken from the index
    assert len(errors) == 3
    assert any("row 3" in w and "gold_page 9" in w for w in warnings)
    assert any("row 4" in w and "own words" in w for w in warnings)


def test_excel_bom_is_handled(index, tmp_path):
    path = tmp_path / "t.csv"
    path.write_text(CSV_HEADER + "Espresso is best with oat milk.,,,,none,\n", encoding="utf-8-sig")
    claims, errors, _ = ev.load_test_set(path, index.table)
    assert len(claims) == 1 and not errors


# ---------------------------------------------------------------- metrics


def test_wilson_interval_known_values():
    lo, hi = ev.wilson(5, 10)
    assert lo == pytest.approx(0.237, abs=0.001) and hi == pytest.approx(0.763, abs=0.001)
    assert ev.wilson(0, 10)[0] == 0.0 and ev.wilson(10, 10)[1] == pytest.approx(1.0)
    assert str(ev.Score(3, 4)).startswith("75% (3/4, 95% CI")


def make_claims(index, tmp_path):
    path = write_csv(tmp_path / "t.csv", [
        f'Noisy bounding boxes reduce detector accuracy.,alpha2020,4,"{GOLD}",supports,',
        'Noisy bounding boxes are common in detection datasets.,alpha2020,1,"Noisy bounding boxes are common in large detection datasets.",related,',
        "Espresso tastes better with oat milk at breakfast.,,,,none,",
        "Coffee improves sleep quality for most adults.,,,,none,",  # this one WILL match gamma2022
    ])
    claims, errors, _ = ev.load_test_set(path, index.table)
    assert not errors
    hits = index.search_many([c.as_claim().query for c in claims], k=10)
    return claims, {c.row: h for c, h in zip(claims, hits)}


def test_retrieval_metrics(index, tmp_path):
    claims, cands = make_claims(index, tmp_path)
    run = ev.run_retrieval("x", index, None, claims, cands, threshold=0.40)
    assert run.ranks[2] is not None and run.ranks[2] <= 3
    assert run.recall(claims, 3).n == 2  # only claims with evidence count
    assert run.no_evidence(claims).short() == "50% (1/2)"  # espresso: none; coffee: found
    assert run.missed(claims).n == 2

    strict = ev.run_retrieval("x", index, None, claims, cands, threshold=1.01)  # above any cosine
    assert strict.no_evidence(claims).k == 2 and strict.missed(claims).k == 2  # nothing found at all


def test_reranker_scores_are_memoised(index, tmp_path):
    claims, cands = make_claims(index, tmp_path)
    inner = KeywordReranker("sharply")
    memo = ev.MemoReranker(inner)
    first = ev.run_retrieval("r", index, memo, claims, cands, 0.30)
    seen = len(inner.pairs_seen)
    ev.run_retrieval("r", index, memo, claims, cands, 0.35)
    assert len(inner.pairs_seen) == seen  # the second run re-used every score
    assert first.ranks[2] == 1  # re-ranking put the gold sentence first


# ---------------------------------------------------------------- labels


class ScriptedLLM:
    """Labels 'supports' unless the claim mentions datasets -> 'supports' too (a mistake)."""

    name, model = "fake", "m"

    def complete(self, system, user):
        return json.dumps({"judgements": [{"id": 1, "label": "supports", "reason": "r"}]})


def test_label_accuracy_confusion_and_precision(index, tmp_path):
    claims, _ = make_claims(index, tmp_path)
    labels = ev.run_labels(Judge(ScriptedLLM(), cache_path=None), claims, index.table, pause=0)
    assert labels.accuracy(claims).short() == "50% (1/2)"  # right on 'supports', wrong on 'related'
    assert labels.precision(claims, "supports").short() == "50% (1/2)"
    confusion = labels.confusion(claims)
    assert confusion["supports"]["supports"] == 1 and confusion["related"]["supports"] == 1


def test_llm_errors_are_recorded_not_raised(index, tmp_path):
    class Down:
        name, model = "fake", "m"

        def complete(self, system, user):
            raise LLMError("HTTP 401: bad key")

    claims, _ = make_claims(index, tmp_path)
    labels = ev.run_labels(Judge(Down(), cache_path=None), claims, index.table, pause=0)
    assert len(labels.errors) == 2 and labels.accuracy(claims).n == 0


# ---------------------------------------------------------------- template + full run


def test_template_samples_evenly_and_adds_none_rows(index, tmp_path):
    out = tmp_path / "template.csv"
    big = pd.DataFrame(
        [(p, i, f"This is sentence number {i} from paper {p} with enough words to be sampled here.")
         for p in ("a", "b", "c") for i in range(20)],
        columns=["paper_id", "page", "sentence"],
    )
    big.insert(0, "sentence_id", range(len(big)))
    big.to_csv(tmp_path / "sentences.csv", index=False)
    ev.make_template(tmp_path, out, n=9, n_none=4, seed=1)
    df = pd.read_csv(out, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert list(df.columns) == ev.COLUMNS
    assert (df["gold_label"] == "none").sum() == 4
    assert df[df["gold_label"] != "none"]["gold_paper"].value_counts().to_dict() == {"a": 3, "b": 3, "c": 3}


def test_full_run_writes_report_and_details(index, tmp_path, monkeypatch):
    claims_csv = tmp_path / "test_claims.csv"
    make_claims(index, tmp_path)  # writes t.csv
    (tmp_path / "t.csv").rename(claims_csv)
    monkeypatch.setattr(EvidenceIndex, "load", classmethod(lambda cls, *a, **k: index))
    monkeypatch.setattr(ev, "load_reranker", lambda: KeywordReranker("sharply"))
    monkeypatch.setattr(Judge, "from_env", classmethod(lambda cls, *a, **k: Judge(ScriptedLLM(), None)))
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path / "results")

    args = argparse.Namespace(index_dir=tmp_path, test_file=claims_csv, no_judge=False,
                              threshold=0.40, pause=0, show_failures=3)
    ev.cmd_run(args)

    report = (tmp_path / "results" / "eval_results.md").read_text(encoding="utf-8")
    assert "Test set: **4 claims**" in report
    assert "| **Recall@3** |" in report and "With re-ranking" in report
    assert "0.40 (current)" in report
    assert "Label accuracy: **50% (1/2" in report
    # Espresso: no evidence (correct). Coffee: evidence found and the fake LLM says supports (wrong).
    assert "No-evidence accuracy (full system): **50% (1/2" in report
    assert "## End to end: the verdict the user sees" in report and "Verdict accuracy, all claims" in report
    assert "0.70" in report
    details = pd.read_csv(tmp_path / "results" / "eval_details.csv", encoding="utf-8-sig")
    assert len(details) == 4 and {"failure_reason", "verdict", "verdict_correct"} <= set(details.columns)
    assert details["failed"].sum() >= 2  # the coffee claim and the mislabelled 'related' claim


def test_verdict_rule_matches_the_pipeline():
    assert ev.verdict_of([]) == "no evidence"
    assert ev.verdict_of(["related", "related"]) == "related"
    assert ev.verdict_of(["related", "supports"]) == "supported"
    assert ev.verdict_of(["contradicts"]) == "contradicted"
    assert ev.verdict_of(["supports", "contradicts"]) == "contested"


def test_verdicts_are_scored_per_gold_label(index, tmp_path):
    class SameLabelLLM:
        """Gives every evidence sentence the same label."""

        name, model = "fake", "m"

        def __init__(self, label):
            self.label = label

        def complete(self, system, user):
            n = user.count("<evidence id=")
            return json.dumps({"judgements": [{"id": i, "label": self.label, "reason": "r"} for i in range(1, n + 1)]})

    claims, cands = make_claims(index, tmp_path)
    run = ev.run_retrieval("r", index, None, claims, cands, threshold=0.40)

    cautious = ev.run_verdicts(Judge(SameLabelLLM("related"), cache_path=None), claims, run, pause=0)
    assert cautious.correct(claims, ("none",)).short() == "100% (2/2)"  # no evidence + only related
    assert cautious.correct(claims, ("supports",)).short() == "0% (0/1)"  # never says supported
    assert cautious.correct(claims, ("related",)).short() == "100% (1/1)"

    eager = ev.run_verdicts(Judge(SameLabelLLM("supports"), cache_path=None), claims, run, pause=0)
    assert eager.correct(claims, ("none",)).short() == "50% (1/2)"  # coffee: falsely supported
    assert eager.correct(claims, ("supports",)).short() == "100% (1/1)"


def test_rerun_keeps_failure_reasons(tmp_path):
    path = tmp_path / "details.csv"
    pd.DataFrame({"row": [2, 3], "claim": ["a", "b"], "failure_reason": ["gold never retrieved", ""]}).to_csv(
        path, index=False, encoding="utf-8-sig")
    new = pd.DataFrame({"row": [2, 3, 4], "claim": ["a", "b", "c"], "failure_reason": ["", "", ""]})
    ev.keep_failure_reasons(new, path)
    assert new["failure_reason"].tolist() == ["gold never retrieved", "", ""]
