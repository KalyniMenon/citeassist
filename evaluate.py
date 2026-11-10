"""Measure how well CiteAssist works on a labelled test set.

Three commands, run from the project root:

  1. python evaluate.py template
       Samples sentences at random from index/sentences.csv into
       data/test_claims_template.csv. For each row you write a claim in your
       own words and its label, then save the file as data/test_claims.csv.
       (Random sampling stops you from unconsciously picking easy cases.)

  2. python evaluate.py check
       Validates data/test_claims.csv: labels, and that every gold sentence
       really exists in the index. Costs no API calls.

  3. python evaluate.py run              (add --no-judge to skip the LLM)
       Runs the test set with and without re-ranking and reports:
         recall@1 / recall@3     is the gold sentence in the top 1 / top 3?
         candidate recall@10     is it anywhere in the 10 FAISS candidates?
         no-evidence accuracy    claims with no evidence -> "No evidence found"?
         verdict accuracy        with the LLM: is the verdict the user sees right?
                                 (supported / contradicted / only related / no evidence)
         missed evidence         claims WITH evidence wrongly given none
         label accuracy          does the LLM label (claim, gold sentence) correctly?
       plus a threshold sweep. Writes results/eval_results.md (paste into the
       README) and results/eval_details.csv (one row per claim, for failure
       analysis).

test_claims.csv columns:
  claim          the claim, in your own words
  gold_paper     paper_id of the evidence sentence (empty for "none")
  gold_page      its page (empty for "none")
  gold_sentence  the evidence sentence, copied from index/sentences.csv (empty for "none")
  gold_label     supports / contradicts / related / none
  notes          optional
"""

from __future__ import annotations

import argparse
import difflib
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.claims import Claim, strip_citations
from src.index import DEFAULT_INDEX_DIR, EvidenceIndex
from src.judge import LABELS, PROMPT_VERSION, Judge, LLMError
from src.retrieve import CANDIDATES, RERANKER_NAME, THRESHOLD, TOP_K, Retriever, load_reranker

ROOT = Path(__file__).resolve().parent
TEST_FILE = ROOT / "data" / "test_claims.csv"
TEMPLATE_FILE = ROOT / "data" / "test_claims_template.csv"
RESULTS_DIR = ROOT / "results"

COLUMNS = ["claim", "gold_paper", "gold_page", "gold_sentence", "gold_label", "notes"]
GOLD_LABELS = (*LABELS, "none")
SWEEP = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


# ---------------------------------------------------------------- test set


@dataclass
class TestClaim:
    row: int  # row number as Excel shows it (the header is row 1)
    claim: str
    gold_label: str
    gold_paper: str = ""
    gold_page: int | None = None
    gold_sentence: str = ""
    gold_id: int | None = None  # sentence_id in index/sentences.csv

    @property
    def has_evidence(self) -> bool:
        return self.gold_label != "none"

    def as_claim(self) -> Claim:
        return Claim(self.row, self.claim, strip_citations(self.claim))


def _norm(text: str) -> str:
    text = text.lower().replace("“", '"').replace("”", '"').replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip().strip("\"'")
    return text.rstrip(".")


def resolve_gold(table: pd.DataFrame, paper: str, sentence: str) -> tuple[int | None, str]:
    """Find the gold sentence in the index table: exact (normalised) match first,
    then a close match within the same paper. Returns (sentence_id, note)."""
    target = _norm(sentence)
    norm = table["sentence"].map(_norm)
    exact = table[norm == target]
    if len(exact):
        same_paper = exact[exact["paper_id"] == paper]
        return int((same_paper if len(same_paper) else exact).iloc[0]["sentence_id"]), ""
    pool = table[table["paper_id"] == paper]
    if pool.empty:
        return None, f"paper '{paper}' has no sentences in the index"
    texts = pool["sentence"].map(_norm).tolist()
    close = difflib.get_close_matches(target, texts, n=1, cutoff=0.85)
    if close:
        sid = int(pool.iloc[texts.index(close[0])]["sentence_id"])
        return sid, "matched approximately (check it was copied correctly)"
    return None, "gold sentence not found in index/sentences.csv"


def load_test_set(path: Path, table: pd.DataFrame) -> tuple[list[TestClaim], list[str], list[str]]:
    """Read and validate test_claims.csv. Returns (claims, errors, warnings)."""
    if not path.exists():
        sys.exit(f"{path} not found. Run `python evaluate.py template` and follow the instructions.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = [c for c in COLUMNS[:-1] if c not in df.columns]
    if missing:
        sys.exit(f"{path} is missing columns: {', '.join(missing)}")

    claims, errors, warnings = [], [], []
    for i, r in enumerate(df.itertuples(index=False), start=2):  # Excel row numbers
        claim = " ".join(r.claim.split())
        label = r.gold_label.strip().lower()
        if not claim and not label:
            continue  # blank row left over from the template
        where = f"row {i}"
        if not claim:
            errors.append(f"{where}: no claim written")
            continue
        if label not in GOLD_LABELS:
            errors.append(f"{where}: gold_label '{r.gold_label}' must be one of {', '.join(GOLD_LABELS)}")
            continue
        tc = TestClaim(i, claim, label)
        if label != "none":
            if not r.gold_sentence.strip():
                errors.append(f"{where}: label '{label}' needs a gold_sentence")
                continue
            tc.gold_paper, tc.gold_sentence = r.gold_paper.strip(), r.gold_sentence.strip()
            tc.gold_id, note = resolve_gold(table, tc.gold_paper, tc.gold_sentence)
            if tc.gold_id is None:
                errors.append(f"{where}: {note}")
                continue
            gold = table.loc[table["sentence_id"] == tc.gold_id].iloc[0]
            tc.gold_page = int(gold["page"])
            if note:
                warnings.append(f"{where}: {note}")
            if r.gold_page.strip() and r.gold_page.strip() != str(tc.gold_page):
                warnings.append(f"{where}: gold_page {r.gold_page} but the index says p. {tc.gold_page} (using the index)")
            if _norm(claim) == _norm(tc.gold_sentence):
                warnings.append(f"{where}: claim is a copy of the gold sentence; write it in your own words")
        claims.append(tc)
    return claims, errors, warnings


# ---------------------------------------------------------------- template


def make_template(index_dir: Path, out: Path, n: int, n_none: int, seed: int) -> None:
    table = pd.read_csv(index_dir / "sentences.csv", dtype={"paper_id": str}, keep_default_na=False)
    words = table["sentence"].str.split().str.len()
    pool = table[(words >= 12) & (words <= 45)]  # long enough to make a claim from, short enough to read
    if len(pool) < n:
        pool = table  # small index: use every sentence
    if pool.empty:
        sys.exit("index/sentences.csv has no sentences. Run `python -m src.ingest` first.")
    papers = sorted(pool["paper_id"].unique())
    per_paper = math.ceil(n / len(papers))
    # The same number from each paper, then shuffled, so no paper dominates.
    per = [pool[pool["paper_id"] == p] for p in papers]
    picks = pd.concat([g.sample(min(per_paper, len(g)), random_state=seed) for g in per])
    picks = picks.sample(frac=1, random_state=seed).head(n)
    rows = [
        {"claim": "", "gold_paper": r.paper_id, "gold_page": r.page, "gold_sentence": r.sentence,
         "gold_label": "", "notes": ""}
        for r in picks.itertuples(index=False)
    ]
    rows += [{c: "" for c in COLUMNS} | {"gold_label": "none"} for _ in range(n_none)]
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False, encoding="utf-8-sig")  # BOM: Excel-friendly
    print(f"Wrote {len(rows)} rows ({len(picks)} sampled sentences from {len(papers)} papers, "
          f"{n_none} 'none' rows) to {out}")
    print("Next: write a claim for each row (see README, 'Evaluation'), delete rows you don't want, "
          f"and save it as {TEST_FILE.relative_to(ROOT)}.")


# ---------------------------------------------------------------- metrics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion (Wilson score)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Score:
    k: int
    n: int

    @property
    def value(self) -> float:
        return self.k / self.n if self.n else float("nan")

    def __str__(self) -> str:
        if not self.n:
            return "n/a"
        lo, hi = wilson(self.k, self.n)
        return f"{self.value:.0%} ({self.k}/{self.n}, 95% CI {lo:.0%}–{hi:.0%})"

    def short(self) -> str:
        return f"{self.value:.0%} ({self.k}/{self.n})" if self.n else "n/a"


class MemoReranker:
    """Remembers cross-encoder scores, so the threshold sweep re-uses them."""

    def __init__(self, inner):
        self.inner, self.memo = inner, {}

    def predict(self, pairs, **kwargs):
        todo = [p for p in pairs if p not in self.memo]
        if todo:
            for p, s in zip(todo, list(self.inner.predict(todo, **kwargs))):
                self.memo[p] = float(s)
        return [self.memo[p] for p in pairs]


@dataclass
class RetrievalRun:
    name: str
    threshold: float
    ranks: dict[int, int | None] = field(default_factory=dict)  # row -> rank of gold in final top-k
    found: dict[int, bool] = field(default_factory=dict)  # row -> any evidence shown?
    top1: dict[int, tuple[str, int, str] | None] = field(default_factory=dict)
    evidence: dict[int, list[str]] = field(default_factory=dict)  # row -> sentences shown

    def recall(self, claims: list[TestClaim], k: int) -> Score:
        ev = [c for c in claims if c.has_evidence]
        return Score(sum(1 for c in ev if (self.ranks[c.row] or 99) <= k), len(ev))

    def no_evidence(self, claims: list[TestClaim]) -> Score:
        none = [c for c in claims if not c.has_evidence]
        return Score(sum(1 for c in none if not self.found[c.row]), len(none))

    def missed(self, claims: list[TestClaim]) -> Score:
        ev = [c for c in claims if c.has_evidence]
        return Score(sum(1 for c in ev if not self.found[c.row]), len(ev))


def run_retrieval(name, index, reranker, claims, candidates_by_row, threshold) -> RetrievalRun:
    retriever = Retriever(index, reranker, candidates=CANDIDATES, top_k=TOP_K, threshold=threshold)
    run = RetrievalRun(name, threshold)
    for c in claims:
        evidence = retriever.select(c.as_claim(), candidates_by_row[c.row])
        run.found[c.row] = bool(evidence)
        run.ranks[c.row] = next((e.rank for e in evidence if e.sentence_id == c.gold_id), None)
        run.top1[c.row] = (evidence[0].paper_id, evidence[0].page, evidence[0].sentence) if evidence else None
        run.evidence[c.row] = [e.sentence for e in evidence]
    return run


@dataclass
class LabelRun:
    predicted: dict[int, tuple[str, str]] = field(default_factory=dict)  # row -> (label, reason)
    errors: dict[int, str] = field(default_factory=dict)

    def accuracy(self, claims: list[TestClaim]) -> Score:
        judged = [c for c in claims if c.row in self.predicted]
        return Score(sum(1 for c in judged if self.predicted[c.row][0] == c.gold_label), len(judged))

    def precision(self, claims: list[TestClaim], label: str) -> Score:
        said = [c for c in claims if self.predicted.get(c.row, ("",))[0] == label]
        return Score(sum(1 for c in said if c.gold_label == label), len(said))

    def confusion(self, claims: list[TestClaim]) -> dict[str, Counter]:
        """gold label -> Counter of predicted labels."""
        m = {l: Counter() for l in LABELS}
        for c in claims:
            if c.row in self.predicted:
                m[c.gold_label][self.predicted[c.row][0]] += 1
        return m


def run_labels(judge: Judge, claims: list[TestClaim], table: pd.DataFrame, pause: float) -> LabelRun:
    """Ask the judge about each (claim, gold sentence) pair, independent of retrieval."""
    run = LabelRun()
    labelled = [c for c in claims if c.gold_label in LABELS]
    for i, c in enumerate(labelled, 1):
        sentence = table.loc[table["sentence_id"] == c.gold_id, "sentence"].iloc[0]
        try:
            j = judge.judge_claim(c.as_claim().query, [sentence])[0]
            run.predicted[c.row] = (j.label, j.reason)
            if j.source == "llm" and pause:
                time.sleep(pause)  # be gentle with free-tier rate limits
        except LLMError as exc:
            run.errors[c.row] = str(exc)
        print(f"\r  labelled {i}/{len(labelled)}", end="", flush=True)
    print()
    return run


# The verdicts that count as right for each gold label. "contested" (both
# supporting and contradicting evidence shown) still surfaces the right evidence.
EXPECTED_VERDICTS = {
    "supports": ("supported", "contested"),
    "contradicts": ("contradicted", "contested"),
    "related": ("related",),
    "none": ("no evidence", "related"),
}


@dataclass
class VerdictRun:
    """End to end: the verdict the user sees for each claim (judge applied to the evidence shown)."""

    verdicts: dict[int, str] = field(default_factory=dict)  # row -> supported / related / no evidence / ...
    errors: dict[int, str] = field(default_factory=dict)

    def is_correct(self, c: TestClaim) -> bool | None:
        v = self.verdicts.get(c.row)
        return None if v is None else v in EXPECTED_VERDICTS[c.gold_label]

    def correct(self, claims: list[TestClaim], gold_labels=GOLD_LABELS) -> Score:
        chosen = [c for c in claims if c.gold_label in gold_labels and c.row in self.verdicts]
        return Score(sum(1 for c in chosen if self.is_correct(c)), len(chosen))


def verdict_of(labels: list[str]) -> str:
    """Same rule as CheckedClaim.verdict in src/pipeline.py."""
    if not labels:
        return "no evidence"
    if "supports" in labels and "contradicts" in labels:
        return "contested"
    if "supports" in labels:
        return "supported"
    if "contradicts" in labels:
        return "contradicted"
    return "related"


def run_verdicts(judge: Judge, claims: list[TestClaim], run: RetrievalRun, pause: float) -> VerdictRun:
    """Judge the evidence actually shown for every claim, as the app would, and record the verdict."""
    out = VerdictRun()
    for i, c in enumerate(claims, 1):
        sentences = run.evidence[c.row]
        if not sentences:
            out.verdicts[c.row] = "no evidence"
        else:
            try:
                judged = judge.judge_claim(c.as_claim().query, sentences)
                out.verdicts[c.row] = verdict_of([j.label for j in judged])
                if pause and any(j.source == "llm" for j in judged):
                    time.sleep(pause)
            except LLMError as exc:
                out.errors[c.row] = str(exc)
        print(f"\r  verdicts {i}/{len(claims)}", end="", flush=True)
    print()
    return out


# ---------------------------------------------------------------- report


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def build_report(claims, main_runs, sweep, labels, settings, cand: Score, verdicts: VerdictRun | None = None) -> str:
    ev = sum(c.has_evidence for c in claims)
    counts = Counter(c.gold_label for c in claims)
    lines = [
        "# CiteAssist evaluation results",
        "",
        f"Test set: **{len(claims)} claims** ({ev} with evidence: {counts['supports']} supports, "
        f"{counts['contradicts']} contradicts, {counts['related']} related; {counts['none']} with no evidence "
        "in the papers).",
        "",
        "Settings: " + ", ".join(f"{k} `{v}`" for k, v in settings.items()) + ".",
        "",
        "## Retrieval",
        "",
        md_table(
            ["Metric", *[r.name for r in main_runs]],
            [
                ["Recall@1", *[r.recall(claims, 1).short() for r in main_runs]],
                ["**Recall@3**", *[f"**{r.recall(claims, 3).short()}**" for r in main_runs]],
                ["No-evidence accuracy (retrieval only)", *[r.no_evidence(claims).short() for r in main_runs]],
                ["Missed evidence (lower is better)", *[r.missed(claims).short() for r in main_runs]],
            ],
        ),
        "",
        f"Candidate recall@{CANDIDATES} (gold sentence anywhere in the FAISS candidates, before the "
        f"threshold): **{cand.short()}**. This is the ceiling for re-ranking.",
        "",
        f"95% confidence intervals with re-ranking: recall@3 {main_runs[-1].recall(claims, 3)}; "
        f"no-evidence accuracy {main_runs[-1].no_evidence(claims)}.",
        "",
        "## Threshold sweep (with re-ranking)",
        "",
        "Raising the minimum similarity gives more correct *No evidence found* answers but misses more real evidence.",
        "",
        md_table(
            ["Threshold", "Recall@3", "No-evidence accuracy", "Missed evidence"],
            [[f"{r.threshold:.2f}" + (" (current)" if abs(r.threshold - settings["threshold"]) < 1e-9 else ""),
              r.recall(claims, 3).short(), r.no_evidence(claims).short(), r.missed(claims).short()] for r in sweep],
        ),
    ]
    if verdicts is not None:
        none_counts = Counter(verdicts.verdicts[c.row] for c in claims if not c.has_evidence and c.row in verdicts.verdicts)
        right = {
            "supports": "Supported (or contested)", "contradicts": "Contradicted (or contested)",
            "related": "Only related evidence", "none": "No evidence, or only related",
        }
        lines += [
            "",
            "## End to end: the verdict the user sees",
            "",
            "The LLM judges the evidence actually shown (top 3 after re-ranking), as in the app. Unlike recall, "
            "this credits *any* shown sentence that supports or contradicts the claim, not only the gold one.",
            "",
            md_table(
                ["Gold label", "Right verdict", "Correct"],
                [[g, right[g], verdicts.correct(claims, (g,)).short()] for g in GOLD_LABELS
                 if any(c.gold_label == g for c in claims)]
                + [["**All claims**", "", f"**{verdicts.correct(claims).short()}**"]],
            ),
            "",
            f"Verdict accuracy, all claims: **{verdicts.correct(claims)}**",
            "",
            f"No-evidence accuracy (full system): **{verdicts.correct(claims, ('none',))}**. On-topic claims that no "
            "paper supports pass the similarity threshold, so it is the judge labelling their evidence *related* that "
            "keeps them from being marked supported. Verdicts given: "
            + ", ".join(f"{v} {n}" for v, n in none_counts.most_common()) + ".",
        ]
        if verdicts.errors:
            lines += ["", f"{len(verdicts.errors)} claims could not be checked (LLM errors); re-run to fill them in."]
    if labels is not None:
        lines += [
            "",
            "## Labels (LLM judge on the gold sentence)",
            "",
            f"Label accuracy: **{labels.accuracy(claims)}**",
            "",
            f"When it says *supports*, it is right {labels.precision(claims, 'supports').short()}; "
            f"*contradicts*: {labels.precision(claims, 'contradicts').short()}.",
            "",
            md_table(
                ["Gold label ↓ / LLM said →", *LABELS],
                [[gold, *[str(row[said]) for said in LABELS]] for gold, row in labels.confusion(claims).items()],
            ),
        ]
        if labels.errors:
            lines += ["", f"{len(labels.errors)} claims could not be labelled (LLM errors). Re-run to fill them in; "
                      "cached answers are not requested again."]
    lines += [
        "",
        "## Notes",
        "",
        "- Recall counts a claim as found only if its *gold* sentence is in the top 3; another sentence that "
        "also supports the claim counts as a miss, so these numbers are a lower bound.",
        "- With this many claims the confidence intervals are wide; differences of a few points are not meaningful.",
        "- Failure analysis: see `results/eval_details.csv` (fill in the `failure_reason` column).",
    ]
    return "\n".join(lines) + "\n"


def details_frame(claims, main_runs, labels, table, cand_ranks, verdicts: VerdictRun | None = None) -> pd.DataFrame:
    sent = table.set_index("sentence_id")["sentence"]
    rows = []
    for c in claims:
        row = {
            "row": c.row, "claim": c.claim, "gold_label": c.gold_label, "gold_paper": c.gold_paper,
            "gold_page": c.gold_page or "", "gold_sentence": sent.get(c.gold_id, "") if c.gold_id is not None else "",
            "candidate_rank": cand_ranks.get(c.row) or "",
        }
        for r in main_runs:
            key = "rerank" if "re-rank" in r.name else "embed"
            row[f"rank_{key}"] = r.ranks[c.row] or ""
            row[f"found_{key}"] = r.found[c.row]
        top = main_runs[-1].top1[c.row]
        row["top1_paper"], row["top1_page"], row["top1_sentence"] = top if top else ("", "", "")
        if labels is not None and c.row in labels.predicted:
            row["judge_label"], row["judge_reason"] = labels.predicted[c.row]
            row["label_correct"] = labels.predicted[c.row][0] == c.gold_label
        else:
            row["judge_label"] = row["judge_reason"] = row["label_correct"] = ""
        rerank = main_runs[-1]
        ok = verdicts.is_correct(c) if verdicts is not None else None
        row["verdict"] = verdicts.verdicts.get(c.row, "") if verdicts is not None else ""
        row["verdict_correct"] = "" if ok is None else ok
        if c.has_evidence:
            failed = (rerank.ranks[c.row] or 99) > 3 or row["label_correct"] is False or ok is False
        elif ok is not None:
            failed = not ok  # false support / contradiction
        else:
            failed = rerank.found[c.row]
        row["failed"] = failed
        row["failure_reason"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def keep_failure_reasons(details: pd.DataFrame, path: Path) -> None:
    """Copy failure_reason values you already wrote into the new details table,
    so re-running the evaluation never erases your notes."""
    if not path.exists():
        return
    old = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "failure_reason" not in old.columns:
        return
    notes = {(r.row, r.claim): r.failure_reason for r in old.itertuples(index=False) if r.failure_reason.strip()}
    details["failure_reason"] = [notes.get((str(r), c), "") for r, c in zip(details["row"], details["claim"])]


def print_failures(df: pd.DataFrame, limit: int) -> None:
    fails = df[df["failed"]]
    print(f"\n{len(fails)} claims had at least one failure. First {min(limit, len(fails))}:")
    for r in fails.head(limit).itertuples(index=False):
        print(f"\n  Row {r.row} [{r.gold_label}]: {r.claim}")
        if r.gold_label != "none":
            where = f"rank {r.rank_rerank}" if r.rank_rerank else (
                f"candidate #{r.candidate_rank} but cut" if r.candidate_rank else "not in the 10 candidates")
            print(f"    gold ({r.gold_paper}, p. {r.gold_page}; {where}): {r.gold_sentence[:110]}")
        if r.verdict:
            print(f"    verdict shown: {r.verdict.upper()} ({'right' if r.verdict_correct is True else 'wrong'})")
        if r.top1_sentence:
            print(f"    top result ({r.top1_paper}, p. {r.top1_page}): {str(r.top1_sentence)[:110]}")
        elif r.gold_label != "none":
            print("    top result: no evidence found")
        if r.label_correct is False:
            print(f"    judge said {r.judge_label.upper()}: {r.judge_reason}")


# ---------------------------------------------------------------- commands


def cmd_check(args) -> list[TestClaim]:
    table = pd.read_csv(args.index_dir / "sentences.csv", dtype={"paper_id": str}, keep_default_na=False)
    claims, errors, warnings = load_test_set(args.test_file, table)
    counts = Counter(c.gold_label for c in claims)
    print(f"{len(claims)} valid claims: " + ", ".join(f"{counts[l]} {l}" for l in GOLD_LABELS))
    for w in warnings:
        print(f"  warning: {w}")
    for e in errors:
        print(f"  ERROR: {e}")
    if errors:
        sys.exit(f"{len(errors)} problem(s) to fix in {args.test_file}.")
    if counts["none"] == 0:
        print("  warning: no 'none' claims, so no-evidence accuracy cannot be measured")
    if len(claims) < 30:
        print("  warning: fewer than 30 claims; results will be very noisy")
    return claims


def cmd_run(args) -> None:
    claims = cmd_check(args)
    print("\nLoading the index and models...")
    index = EvidenceIndex.load(args.index_dir)
    reranker = MemoReranker(load_reranker())
    judge = None if args.no_judge else Judge.from_env()

    tests = [c.as_claim() for c in claims]
    hits = index.search_many([t.query for t in tests], k=CANDIDATES)
    candidates_by_row = {c.row: h for c, h in zip(claims, hits)}
    cand_ranks = {c.row: next((i for i, h in enumerate(candidates_by_row[c.row], 1) if h.sentence_id == c.gold_id), None)
                  for c in claims}
    ev = [c for c in claims if c.has_evidence]
    cand = Score(sum(1 for c in ev if cand_ranks[c.row]), len(ev))

    print("Retrieval: embeddings only, and with re-ranking...")
    main_runs = [
        run_retrieval("Embeddings only", index, None, claims, candidates_by_row, args.threshold),
        run_retrieval("With re-ranking", index, reranker, claims, candidates_by_row, args.threshold),
    ]
    sweep = [run_retrieval("sweep", index, reranker, claims, candidates_by_row, t)
             for t in sorted(set(SWEEP) | {args.threshold})]

    labels = verdicts = None
    if judge is not None:
        print(f"Labels: {judge.provider.name} / {judge.provider.model}, prompt {PROMPT_VERSION}...")
        labels = run_labels(judge, claims, index.table, args.pause)
        print("End-to-end verdicts (judging the evidence shown)...")
        verdicts = run_verdicts(judge, claims, main_runs[-1], args.pause)

    settings = {
        "embedding model": index.meta.get("model"), "re-ranker": RERANKER_NAME,
        "LLM": f"{judge.provider.name} / {judge.provider.model}" if judge else "off",
        "prompt": PROMPT_VERSION, "threshold": args.threshold,
        "date": time.strftime("%Y-%m-%d"),
    }
    report = build_report(claims, main_runs, sweep, labels, settings, cand, verdicts)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "eval_results.md").write_text(report, encoding="utf-8")
    details = details_frame(claims, main_runs, labels, index.table, cand_ranks, verdicts)
    keep_failure_reasons(details, RESULTS_DIR / "eval_details.csv")
    details.to_csv(RESULTS_DIR / "eval_details.csv", index=False, encoding="utf-8-sig")

    print("\n" + report)
    print_failures(details, args.show_failures)
    print(f"\nSaved {RESULTS_DIR / 'eval_results.md'} and {RESULTS_DIR / 'eval_details.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CiteAssist on a labelled test set.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--test-file", type=Path, default=TEST_FILE)
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("template", help="sample sentences into a test-set template")
    t.add_argument("--n", type=int, default=60, help="sentences to sample (default 60; keep the best ~40)")
    t.add_argument("--none", type=int, default=12, help="blank rows for no-evidence claims (default 12)")
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--out", type=Path, default=TEMPLATE_FILE)

    sub.add_parser("check", help="validate data/test_claims.csv")

    r = sub.add_parser("run", help="run the evaluation")
    r.add_argument("--no-judge", action="store_true", help="skip the LLM (retrieval metrics only)")
    r.add_argument("--threshold", type=float, default=THRESHOLD)
    r.add_argument("--pause", type=float, default=2.0, help="seconds between LLM calls (default 2)")
    r.add_argument("--show-failures", type=int, default=8)

    args = parser.parse_args()
    if args.command == "template":
        if args.out.resolve() == TEST_FILE.resolve():
            sys.exit("Refusing to overwrite data/test_claims.csv; choose another --out.")
        make_template(args.index_dir, args.out, args.n, args.none, args.seed)
    elif args.command == "check":
        cmd_check(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
