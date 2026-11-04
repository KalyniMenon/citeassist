"""Day 2: embed every sentence and build a FAISS search index.

Build (after running src.ingest):
    python -m src.index build

Search from the terminal:
    python -m src.index search "Label noise reduces detector accuracy." -k 5
    python -m src.index search "claim one" "claim two"

Use from Python (day 3 onwards):
    from src.index import EvidenceIndex
    idx = EvidenceIndex.load()
    hits = idx.search("Missing annotations hurt recall.", k=5)

How it works:
  - Each sentence becomes a 384-number vector (an "embedding") from the
    all-MiniLM-L6-v2 model. Sentences with similar meaning get similar vectors.
  - Vectors are normalised to length 1, so the inner product of two vectors
    equals their cosine similarity (1 = same meaning, ~0 = unrelated).
  - FAISS IndexFlatIP compares the claim's vector with every sentence vector
    (exact search). With a few thousand sentences this takes milliseconds,
    so no approximate index is needed.

Files written to index/:
  faiss.index      the vectors
  index_meta.json  model name, sentence count and a fingerprint of the
                   sentence table, used to detect a stale index
  (sentences.csv is written by src.ingest; row i <-> vector i)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = ROOT / "index"
SENTENCES_FILE = "sentences.csv"
INDEX_FILE = "faiss.index"
META_FILE = "index_meta.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Encoder(Protocol):
    """Anything with a sentence-transformers style encode() method."""

    def encode(self, sentences: list[str], **kwargs) -> np.ndarray: ...


_MODEL_CACHE: dict[str, Encoder] = {}


def load_model(name: str = MODEL_NAME) -> Encoder:
    """Load (and cache) the embedding model. Downloads ~90 MB the first time."""
    if name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer  # slow import, so done lazily

        _MODEL_CACHE[name] = SentenceTransformer(name, device="cpu")
    return _MODEL_CACHE[name]


def embed(model: Encoder, texts: list[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
    """Embed texts as unit-length float32 vectors, shape (len(texts), dim)."""
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=progress,
    )
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    # Normalise again in case a custom encoder ignored normalize_embeddings.
    faiss.normalize_L2(vectors)
    return vectors


def fingerprint(sentences: pd.Series) -> str:
    """Short hash of the sentence texts, to notice when they change."""
    h = hashlib.sha1()
    for s in sentences:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def read_sentences(index_dir: Path) -> pd.DataFrame:
    path = index_dir / SENTENCES_FILE
    if not path.exists():
        sys.exit(f"{path} not found. Run `python -m src.ingest` first.")
    table = pd.read_csv(path, dtype={"paper_id": str, "sentence": str}, keep_default_na=False)
    expected = ["sentence_id", "paper_id", "page", "sentence"]
    if list(table.columns) != expected:
        sys.exit(f"{path} has columns {list(table.columns)}, expected {expected}. Re-run src.ingest.")
    if table.empty:
        sys.exit(f"{path} has no sentences. Check data/papers/ and re-run src.ingest.")
    return table


# ---------------------------------------------------------------- build


def build_index(
    index_dir: Path = DEFAULT_INDEX_DIR,
    model: Encoder | None = None,
    model_name: str = MODEL_NAME,
    batch_size: int = 64,
    progress: bool = True,
) -> "EvidenceIndex":
    """Embed index/sentences.csv and save the FAISS index next to it."""
    table = read_sentences(index_dir)
    model = model if model is not None else load_model(model_name)

    start = time.perf_counter()
    vectors = embed(model, table["sentence"].tolist(), batch_size=batch_size, progress=progress)
    seconds = time.perf_counter() - start

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(index_dir / INDEX_FILE))
    meta = {
        "model": model_name,
        "dim": int(vectors.shape[1]),
        "count": int(index.ntotal),
        "fingerprint": fingerprint(table["sentence"]),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (index_dir / META_FILE).write_text(json.dumps(meta, indent=2))

    print(
        f"Indexed {index.ntotal} sentences from {table['paper_id'].nunique()} papers "
        f"({vectors.shape[1]}-dim vectors) in {seconds:.1f}s -> {index_dir / INDEX_FILE}"
    )
    return EvidenceIndex(index, table, model, meta)


# ---------------------------------------------------------------- search


@dataclass
class Hit:
    rank: int
    score: float  # cosine similarity, -1..1 (higher = closer in meaning)
    sentence_id: int
    paper_id: str
    page: int
    sentence: str


class EvidenceIndex:
    """The saved index plus its sentence table, ready to search."""

    def __init__(self, index: faiss.Index, table: pd.DataFrame, model: Encoder, meta: dict):
        self.index = index
        self.table = table.reset_index(drop=True)
        self.model = model
        self.meta = meta

    @classmethod
    def load(cls, index_dir: Path = DEFAULT_INDEX_DIR, model: Encoder | None = None) -> "EvidenceIndex":
        """Load the index from disk without re-embedding anything."""
        index_path, meta_path = index_dir / INDEX_FILE, index_dir / META_FILE
        if not index_path.exists() or not meta_path.exists():
            sys.exit(f"No index in {index_dir}. Run `python -m src.index build` first.")
        meta = json.loads(meta_path.read_text())
        table = read_sentences(index_dir)
        index = faiss.read_index(str(index_path))

        if index.ntotal != len(table) or meta.get("fingerprint") != fingerprint(table["sentence"]):
            sys.exit(
                "index/sentences.csv has changed since the index was built "
                f"({index.ntotal} vectors, {len(table)} sentences). "
                "Run `python -m src.index build` again."
            )
        model = model if model is not None else load_model(meta["model"])
        return cls(index, table, model, meta)

    def __len__(self) -> int:
        return self.index.ntotal

    def search(self, claim: str, k: int = 5) -> list[Hit]:
        """Return the k sentences closest in meaning to the claim."""
        return self.search_many([claim], k)[0]

    def search_many(self, claims: list[str], k: int = 5) -> list[list[Hit]]:
        """Search several claims in one batch (faster than one at a time)."""
        if not claims:
            return []
        k = max(1, min(k, len(self)))
        scores, ids = self.index.search(embed(self.model, claims), k)
        results = []
        for row_scores, row_ids in zip(scores, ids):
            hits = []
            for rank, (score, i) in enumerate(zip(row_scores, row_ids), start=1):
                if i < 0:  # FAISS pads with -1 when there are fewer than k vectors
                    continue
                r = self.table.iloc[int(i)]
                hits.append(
                    Hit(rank, float(score), int(r.sentence_id), r.paper_id, int(r.page), r.sentence)
                )
            results.append(hits)
        return results


# ---------------------------------------------------------------- CLI


def print_hits(claim: str, hits: list[Hit], width: int = 110) -> None:
    print(f"\nClaim: {claim}")
    for h in hits:
        text = h.sentence if len(h.sentence) <= width else h.sentence[: width - 1] + "…"
        print(f"  {h.rank}. [{h.score:.3f}] {h.paper_id}, p. {h.page}")
        print(f"     {text}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="embed index/sentences.csv and save the index")
    b.add_argument("--batch-size", type=int, default=64)

    s = sub.add_parser("search", help="search one or more claims")
    s.add_argument("claims", nargs="+", help="claim text (quote each one)")
    s.add_argument("-k", type=int, default=5, help="results per claim (default 5)")

    args = parser.parse_args()
    if args.command == "build":
        build_index(args.index_dir, batch_size=args.batch_size)
    else:
        idx = EvidenceIndex.load(args.index_dir)
        for claim, hits in zip(args.claims, idx.search_many(args.claims, args.k)):
            print_hits(claim, hits)


if __name__ == "__main__":
    main()
