"""Find the best evidence sentences for each claim in a paragraph.

For each claim:
  1. Get the top 10 candidate sentences from the FAISS index (src.index).
  2. Drop candidates whose cosine similarity is below a threshold. If none
     are left, the claim gets "no evidence found" instead of a weak match.
  3. Optionally re-rank the survivors with a cross-encoder, which reads the
     claim and the sentence together and judges relevance more precisely
     than comparing two separate embeddings.
  4. Keep at most 2 sentences per paper, so the evidence comes from several
     papers where possible, and return the best 3.

Run from the project root:
    python -m src.retrieve "Your paragraph here. It can have several sentences."
    python -m src.retrieve --file paragraph.txt
    python -m src.retrieve --no-rerank "..."          # embeddings only
    python -m src.retrieve --compare "..."            # with vs without re-ranking

From Python (used by app.py and evaluate.py):
    from src.retrieve import Retriever
    retriever = Retriever.load()
    for result in retriever.retrieve_paragraph(paragraph):
        print(result.claim.text, result.found)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from src.claims import Claim, split_claims
from src.index import DEFAULT_INDEX_DIR, EvidenceIndex, Hit

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Defaults. The 0.40 threshold was set from early checks (real matches scored
# 0.64-0.86, an off-topic claim 0.15) and deliberately NOT re-tuned on the test
# set. The evaluation shows recall@3 is flat from 0.30 to 0.55, and that on-topic
# claims with no evidence score 0.58-0.73, so it is the LLM judge (labelling their
# evidence 'related'), not the threshold, that handles them. See results/.
CANDIDATES = 10
TOP_K = 3
THRESHOLD = 0.40
MAX_PER_PAPER = 2


class Reranker(Protocol):
    """Anything with a sentence-transformers CrossEncoder style predict()."""

    def predict(self, pairs: list[tuple[str, str]], **kwargs) -> np.ndarray: ...


_RERANKER_CACHE: dict[str, Reranker] = {}


def load_reranker(name: str = RERANKER_NAME) -> Reranker:
    """Load (and cache) the cross-encoder. Downloads ~90 MB the first time."""
    if name not in _RERANKER_CACHE:
        import torch
        from sentence_transformers import CrossEncoder  # slow import, so done lazily

        # Return raw logits; select() turns them into 0..1 scores with a
        # sigmoid, so the scale never depends on the model's own config.
        _RERANKER_CACHE[name] = CrossEncoder(name, device="cpu", activation_fn=torch.nn.Identity())
    return _RERANKER_CACHE[name]


@dataclass
class Evidence:
    rank: int
    paper_id: str
    page: int
    sentence: str
    sentence_id: int
    similarity: float  # cosine similarity from the embedding search
    rerank_score: float | None = None  # cross-encoder relevance, 0..1 (None if not re-ranked)


@dataclass
class ClaimResult:
    claim: Claim
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.evidence)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Retriever:
    def __init__(
        self,
        index: EvidenceIndex,
        reranker: Reranker | None = None,
        candidates: int = CANDIDATES,
        top_k: int = TOP_K,
        threshold: float = THRESHOLD,
        max_per_paper: int | None = MAX_PER_PAPER,
    ):
        self.index = index
        self.reranker = reranker
        self.candidates = candidates
        self.top_k = top_k
        self.threshold = threshold
        self.max_per_paper = max_per_paper

    @classmethod
    def load(cls, index_dir: Path = DEFAULT_INDEX_DIR, rerank: bool = True, **settings) -> "Retriever":
        return cls(EvidenceIndex.load(index_dir), load_reranker() if rerank else None, **settings)

    # -------------------------------------------------------------- core

    def retrieve_paragraph(self, paragraph: str) -> list[ClaimResult]:
        return self.retrieve_claims(split_claims(paragraph))

    def retrieve_claims(self, claims: list[Claim]) -> list[ClaimResult]:
        if not claims:
            return []
        hit_lists = self.index.search_many([c.query for c in claims], k=self.candidates)
        return [ClaimResult(c, self.select(c, hits)) for c, hits in zip(claims, hit_lists)]

    def retrieve(self, claim_text: str) -> ClaimResult:
        """Retrieve evidence for one claim given as plain text."""
        return self.retrieve_claims([Claim(1, claim_text, claim_text)])[0]

    def select(self, claim: Claim, hits: list[Hit]) -> list[Evidence]:
        """Apply threshold, re-ranking and per-paper cap to candidate hits (steps 2-4)."""
        # Step 2: the threshold uses the embedding similarity, which is
        # comparable across claims; it decides IF there is evidence.
        hits = [h for h in hits if h.score >= self.threshold]
        if not hits:
            return []

        # Step 3: the cross-encoder only decides the ORDER of what passed.
        rerank: list[float | None] = [None] * len(hits)
        if self.reranker is not None:
            logits = self.reranker.predict([(claim.query, h.sentence) for h in hits], show_progress_bar=False)
            rerank = [_sigmoid(float(x)) for x in np.asarray(logits).reshape(-1)]
            order = sorted(range(len(hits)), key=lambda i: rerank[i], reverse=True)
            hits, rerank = [hits[i] for i in order], [rerank[i] for i in order]

        # Step 4: spread evidence across papers, then keep the best top_k.
        chosen: list[Evidence] = []
        per_paper: dict[str, int] = {}
        for h, r in zip(hits, rerank):
            if self.max_per_paper is not None and per_paper.get(h.paper_id, 0) >= self.max_per_paper:
                continue
            per_paper[h.paper_id] = per_paper.get(h.paper_id, 0) + 1
            chosen.append(Evidence(len(chosen) + 1, h.paper_id, h.page, h.sentence, h.sentence_id, h.score, r))
            if len(chosen) == self.top_k:
                break
        return chosen


# ---------------------------------------------------------------- CLI


def print_results(results: list[ClaimResult], width: int = 110) -> None:
    for res in results:
        print(f"\nClaim {res.claim.index}: {res.claim.text}")
        if not res.found:
            print("  No evidence found in the indexed papers.")
            continue
        for e in res.evidence:
            score = f"sim {e.similarity:.3f}"
            if e.rerank_score is not None:
                score += f", rerank {e.rerank_score:.3f}"
            text = e.sentence if len(e.sentence) <= width else e.sentence[: width - 1] + "…"
            print(f"  {e.rank}. {e.paper_id}, p. {e.page}  [{score}]")
            print(f"     {text}")


def print_comparison(plain: list[ClaimResult], reranked: list[ClaimResult], width: int = 100) -> None:
    for a, b in zip(plain, reranked):
        print(f"\nClaim {a.claim.index}: {a.claim.text}")
        for label, res in (("Embeddings only", a), ("With re-ranking", b)):
            print(f"  {label}:")
            if not res.found:
                print("    No evidence found.")
            for e in res.evidence:
                text = e.sentence if len(e.sentence) <= width else e.sentence[: width - 1] + "…"
                print(f"    {e.rank}. {e.paper_id}, p. {e.page}: {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paragraph", nargs="?", help="the paragraph to check (in quotes)")
    parser.add_argument("--file", type=Path, help="read the paragraph from a text file instead")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")
    parser.add_argument("--compare", action="store_true", help="show results with and without re-ranking")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--candidates", type=int, default=CANDIDATES)
    parser.add_argument("--max-per-paper", type=int, default=MAX_PER_PAPER, help="0 = no limit")
    args = parser.parse_args()

    if args.file:
        paragraph = args.file.read_text(encoding="utf-8")
    elif args.paragraph:
        paragraph = args.paragraph
    else:
        parser.error("give a paragraph in quotes, or --file path/to/paragraph.txt")

    settings = dict(
        candidates=args.candidates,
        top_k=args.top_k,
        threshold=args.threshold,
        max_per_paper=args.max_per_paper or None,
    )
    claims = split_claims(paragraph)
    if not claims:
        print("No claims found (sentences need at least 4 words).")
        return

    index = EvidenceIndex.load(args.index_dir)
    if args.compare:
        plain = Retriever(index, None, **settings).retrieve_claims(claims)
        reranked = Retriever(index, load_reranker(), **settings).retrieve_claims(claims)
        print_comparison(plain, reranked)
    else:
        reranker = None if args.no_rerank else load_reranker()
        print_results(Retriever(index, reranker, **settings).retrieve_claims(claims))


if __name__ == "__main__":
    main()
