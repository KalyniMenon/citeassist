# CiteAssist evaluation results

Test set: **49 claims** (37 with evidence: 27 supports, 7 contradicts, 3 related; 12 with no evidence in the papers).

Settings: embedding model `sentence-transformers/all-MiniLM-L6-v2`, re-ranker `cross-encoder/ms-marco-MiniLM-L-6-v2`, LLM `gemini / gemini-3.5-flash-lite`, prompt `v3`, threshold `0.4`, date `2026-11-09`.

## Retrieval

| Metric | Embeddings only | With re-ranking |
|---|---|---|
| Recall@1 | 30% (11/37) | 32% (12/37) |
| **Recall@3** | **41% (15/37)** | **54% (20/37)** |
| No-evidence accuracy (retrieval only) | 0% (0/12) | 0% (0/12) |
| Missed evidence (lower is better) | 0% (0/37) | 0% (0/37) |

Candidate recall@10 (gold sentence anywhere in the FAISS candidates, before the threshold): **68% (25/37)**. This is the ceiling for re-ranking.

95% confidence intervals with re-ranking: recall@3 54% (20/37, 95% CI 38%–69%); no-evidence accuracy 0% (0/12, 95% CI 0%–24%).

## Threshold sweep (with re-ranking)

Raising the minimum similarity gives more correct *No evidence found* answers but misses more real evidence.

| Threshold | Recall@3 | No-evidence accuracy | Missed evidence |
|---|---|---|---|
| 0.30 | 54% (20/37) | 0% (0/12) | 0% (0/37) |
| 0.35 | 54% (20/37) | 0% (0/12) | 0% (0/37) |
| 0.40 (current) | 54% (20/37) | 0% (0/12) | 0% (0/37) |
| 0.45 | 54% (20/37) | 0% (0/12) | 0% (0/37) |
| 0.50 | 54% (20/37) | 0% (0/12) | 0% (0/37) |
| 0.55 | 54% (20/37) | 0% (0/12) | 0% (0/37) |
| 0.60 | 49% (18/37) | 8% (1/12) | 3% (1/37) |
| 0.65 | 51% (19/37) | 25% (3/12) | 8% (3/37) |
| 0.70 | 43% (16/37) | 92% (11/12) | 24% (9/37) |

## End to end: the verdict the user sees

The LLM judges the evidence actually shown (top 3 after re-ranking), as in the app. Unlike recall, this credits *any* shown sentence that supports or contradicts the claim, not only the gold one.

| Gold label | Right verdict | Correct |
|---|---|---|
| supports | Supported (or contested) | 78% (21/27) |
| contradicts | Contradicted (or contested) | 71% (5/7) |
| related | Only related evidence | 67% (2/3) |
| none | No evidence, or only related | 100% (12/12) |
| **All claims** |  | **82% (40/49)** |

Verdict accuracy, all claims: **82% (40/49, 95% CI 69%–90%)**

No-evidence accuracy (full system): **100% (12/12, 95% CI 76%–100%)**. On-topic claims that no paper supports pass the similarity threshold, so it is the judge labelling their evidence *related* that keeps them from being marked supported. Verdicts given: related 12.

## Labels (LLM judge on the gold sentence)

Label accuracy: **97% (36/37, 95% CI 86%–100%)**

When it says *supports*, it is right 100% (26/26); *contradicts*: 100% (7/7).

| Gold label ↓ / LLM said → | supports | contradicts | related |
|---|---|---|---|
| supports | 26 | 0 | 1 |
| contradicts | 0 | 7 | 0 |
| related | 0 | 0 | 3 |

## Notes

- Recall counts a claim as found only if its *gold* sentence is in the top 3; another sentence that also supports the claim counts as a miss, so these numbers are a lower bound.
- With this many claims the confidence intervals are wide; differences of a few points are not meaningful.
- Failure analysis: see `results/eval_details.csv` (fill in the `failure_reason` column).
