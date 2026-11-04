# CiteAssist

AI research assistant that finds evidence in research papers for each claim in a paragraph and suggests APA citations.

> 🚧 Work in progress — week 1 build. Full README (demo GIF, pipeline, results) comes on day 7.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env               # add your LLM API key on day 4
```

## Day 1: ingest papers

1. Download each paper listed in `data/papers.csv` (the `doi_or_url` column) and save it in `data/papers/` under the name in the `file` column, e.g. `li2020.pdf`. Add rows for any extra papers (aim for 10–20).
2. Run:

```bash
python -m src.ingest
```

This writes `index/sentences.csv` (one row per clean sentence: `sentence_id`, `paper_id`, `page`, `sentence`) and prints a per-paper summary. Check the summary: every paper should say where its references start, and have a few hundred sentences.

What the cleaning does: removes running headers/footers, page numbers, arXiv stamps, emails/URLs, tables and the reference list; fixes hyphenated line breaks; stops section headings merging into sentences; keeps sentences that cross a page break whole (tagged with the page they start on).

## Day 2: build the search index

```bash
python -m src.index build
```

This embeds every sentence in `index/sentences.csv` with [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (downloaded once, ~90 MB) and saves a FAISS `IndexFlatIP` index to `index/faiss.index`. Vectors are normalised, so each score is the cosine similarity between the claim and a sentence.

Search from the terminal:

```bash
python -m src.index search "Missing annotations have a mild effect on detector accuracy." -k 5
```

Or from Python:

```python
from src.index import EvidenceIndex
idx = EvidenceIndex.load()          # loads from disk, no re-embedding
for hit in idx.search("Label noise reduces detection accuracy.", k=5):
    print(hit.score, hit.paper_id, hit.page, hit.sentence)
```

If you re-run `src.ingest`, run `src.index build` again; loading a stale index stops with a message saying so.

## Day 3: from paragraph to evidence

```bash
python -m src.retrieve "Faster R-CNN is fairly robust to missing annotations. Noisy boxes hurt accuracy (Zhou et al., 2023)."
python -m src.retrieve --file paragraph.txt
python -m src.retrieve --compare "..."      # embeddings only vs with re-ranking
python -m src.retrieve --no-rerank "..."    # embeddings only
```

For each claim (one per sentence; citation markers such as `[3]` or `(Wu et al., 2018)` are removed before searching):

1. take the top 10 candidates from the FAISS index;
2. drop candidates with cosine similarity below **0.40**, so a claim with nothing relevant gets *"No evidence found"* instead of a weak match;
3. re-rank the rest with the cross-encoder [ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2), which reads claim and sentence together (downloaded once, ~90 MB);
4. keep at most 2 sentences per paper and return the best 3.

All of these are options: `--threshold`, `--candidates`, `--top-k`, `--max-per-paper` (0 = no limit).

## Tests

```bash
python -m pytest -q
```

`tests/test_ingest.py` builds a deliberately messy two-column PDF and checks each cleaning rule. `tests/test_index.py` checks building, saving, reloading and ranking, using a small word-count encoder in place of MiniLM so the tests run offline in about a second. `tests/test_claims.py` and `tests/test_retrieve.py` cover claim splitting, citation stripping, the threshold, re-ranking and the per-paper cap.

## Project structure

```
citeassist/
├── data/
│   ├── papers/            # the PDF files (not committed)
│   ├── papers.csv         # metadata for each paper
│   └── test_claims.csv    # evaluation test set (day 6)
├── index/                 # saved FAISS index + sentence table (generated)
├── src/
│   ├── ingest.py          # read PDFs, split into sentences
│   ├── index.py           # embed sentences, build/load FAISS index, search
│   ├── claims.py          # split a paragraph into claims
│   ├── retrieve.py        # threshold, re-ranking, per-paper cap
│   ├── judge.py           # day 4
│   └── cite.py            # day 4
├── evaluate.py            # day 6
└── app.py                 # day 5
```
