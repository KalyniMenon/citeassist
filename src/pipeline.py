"""Day 4: the full CiteAssist pipeline in one place.

    paragraph -> claims -> evidence (retrieve) -> labels (judge) -> citations (cite)

Run from the project root:
    python -m src.pipeline "Your paragraph here."
    python -m src.pipeline --file paragraph.txt
    python -m src.pipeline --no-judge "..."     # skip the LLM (no API calls)
    python -m src.pipeline --no-rerank "..."

From Python (the app on day 5, the evaluation on day 6):
    from src.pipeline import CiteAssist
    assistant = CiteAssist.load()
    for checked in assistant.check(paragraph):
        ...
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.cite import DEFAULT_META, Paper, in_text, load_papers, reference
from src.claims import Claim, split_claims
from src.index import DEFAULT_INDEX_DIR
from src.judge import DEFAULT_CACHE, Judge, Judgement, LLMError
from src.retrieve import THRESHOLD, Evidence, Retriever

# Order used when showing evidence: the most useful labels first.
LABEL_ORDER = {"supports": 0, "contradicts": 1, "related": 2}


@dataclass
class CheckedEvidence:
    evidence: Evidence
    judgement: Judgement | None  # None when the judge is switched off
    paper: Paper
    citation: str  # in-text, e.g. (Li et al., 2020, p. 5)

    @property
    def label(self) -> str | None:
        return self.judgement.label if self.judgement else None


@dataclass
class CheckedClaim:
    claim: Claim
    items: list[CheckedEvidence] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.items)

    @property
    def verdict(self) -> str:
        """One word for the whole claim: supported / contested / contradicted / related / no evidence."""
        labels = {i.label for i in self.items}
        if not self.items:
            return "no evidence"
        if "supports" in labels and "contradicts" in labels:
            return "contested"
        if "supports" in labels:
            return "supported"
        if "contradicts" in labels:
            return "contradicted"
        return "related" if None not in labels else "found"


class CiteAssist:
    def __init__(self, retriever: Retriever, judge: Judge | None, papers: dict[str, Paper]):
        self.retriever = retriever
        self.judge = judge
        self.papers = papers

    @classmethod
    def load(
        cls,
        index_dir: Path = DEFAULT_INDEX_DIR,
        meta_path: Path = DEFAULT_META,
        rerank: bool = True,
        use_judge: bool = True,
        cache_path: Path | None = DEFAULT_CACHE,
        **retriever_settings,
    ) -> "CiteAssist":
        retriever = Retriever.load(index_dir, rerank=rerank, **retriever_settings)
        judge = Judge.from_env(cache_path) if use_judge else None
        return cls(retriever, judge, load_papers(meta_path))

    def check(self, paragraph: str) -> list[CheckedClaim]:
        return self.check_claims(split_claims(paragraph))

    def check_claims(self, claims: list[Claim]) -> list[CheckedClaim]:
        checked = []
        for result in self.retriever.retrieve_claims(claims):
            judgements: list[Judgement | None] = [None] * len(result.evidence)
            if self.judge is not None and result.evidence:
                judgements = self.judge.judge_claim(result.claim.query, [e.sentence for e in result.evidence])
            items = []
            for ev, j in zip(result.evidence, judgements):
                paper = self.papers.get(ev.paper_id)
                if paper is None:
                    sys.exit(f"Paper {ev.paper_id!r} is in the index but not in papers.csv.")
                items.append(CheckedEvidence(ev, j, paper, in_text(paper, ev.page)))
            # Supports first, then contradicts, then related; keep retrieval order within a label.
            items.sort(key=lambda i: (LABEL_ORDER.get(i.label or "", 3), i.evidence.rank))
            checked.append(CheckedClaim(result.claim, items))
        return checked


# ---------------------------------------------------------------- CLI


def print_checked(checked: list[CheckedClaim], papers_used: dict[str, Paper]) -> None:
    for c in checked:
        print(f"\nClaim {c.claim.index} [{c.verdict.upper()}]: {c.claim.text}")
        if not c.found:
            print("  No evidence found in the indexed papers.")
        for i, item in enumerate(c.items, 1):
            label = item.label.upper() if item.label else "EVIDENCE"
            print(f"  {i}. {label} {item.citation}")
            print(f'     "{item.evidence.sentence}"')
            if item.judgement:
                print(f"     Why: {item.judgement.reason}")
            papers_used[item.paper.paper_id] = item.paper
    if papers_used:
        print("\nReferences")
        for paper in sorted(papers_used.values(), key=lambda p: (p.authors[0][0] if p.authors else "", p.year)):
            print(f"  {reference(paper)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paragraph", nargs="?")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    if args.file:
        paragraph = args.file.read_text(encoding="utf-8")
    elif args.paragraph:
        paragraph = args.paragraph
    else:
        parser.error("give a paragraph in quotes, or --file path/to/paragraph.txt")

    assistant = CiteAssist.load(
        rerank=not args.no_rerank, use_judge=not args.no_judge, threshold=args.threshold
    )
    if assistant.judge:
        print(f"LLM: {assistant.judge.provider.name} / {assistant.judge.provider.model}")
    try:
        checked = assistant.check(paragraph)
    except LLMError as exc:
        sys.exit(f"LLM error: {exc}\nTip: run with --no-judge to see the evidence without labels.")
    print_checked(checked, {})


if __name__ == "__main__":
    main()
