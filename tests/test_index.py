"""Tests for src/index.py.

They use a tiny bag-of-words encoder instead of the real MiniLM model, so they
run offline in about a second. The real model is checked by hand with
`python -m src.index search ...` (see README, day 2).

Run: python -m pytest -q
"""

import re
import zlib

import numpy as np
import pandas as pd
import pytest

from src.index import EvidenceIndex, build_index, embed

SENTENCES = [
    ("li2020", 3, "Noisy bounding box annotations cause a large drop in detection accuracy."),
    ("li2020", 4, "We propose a cleaning step that corrects noisy class labels before training."),
    ("xu2019", 2, "Missing annotations are treated as background and confuse the detector."),
    ("xu2019", 5, "Training with image level labels recovers most of the lost recall."),
    ("wu2018", 6, "Soft sampling lowers the weight of ambiguous background regions."),
    ("wu2018", 7, "Faster R-CNN is fairly robust when up to half of the labels are missing."),
]


class BagOfWords:
    """Stand-in encoder: counts hashed words into a fixed-size vector."""

    dim = 512

    def __init__(self):
        self.calls = 0

    def encode(self, sentences, **kwargs):
        self.calls += 1
        out = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for row, text in enumerate(sentences):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[row, zlib.crc32(word.encode()) % self.dim] += 1.0
        return out


@pytest.fixture()
def index_dir(tmp_path):
    table = pd.DataFrame(SENTENCES, columns=["paper_id", "page", "sentence"])
    table.insert(0, "sentence_id", range(len(table)))
    table.to_csv(tmp_path / "sentences.csv", index=False)
    return tmp_path


@pytest.fixture()
def built(index_dir):
    return build_index(index_dir, model=BagOfWords(), progress=False)


def test_build_writes_files_with_one_vector_per_sentence(index_dir, built):
    assert (index_dir / "faiss.index").exists()
    assert (index_dir / "index_meta.json").exists()
    assert len(built) == len(SENTENCES)


def test_vectors_are_unit_length():
    vectors = embed(BagOfWords(), ["some words here", "other words"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_best_match_comes_first_with_its_paper_and_page(built):
    hits = built.search("missing annotations confuse the detector", k=3)
    top = hits[0]
    assert (top.paper_id, top.page) == ("xu2019", 2)
    assert top.sentence.startswith("Missing annotations")
    assert [h.rank for h in hits] == [1, 2, 3]


def test_scores_are_cosine_similarities_in_descending_order(built):
    scores = [h.score for h in built.search("noisy labels", k=6)]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0001 <= s <= 1.0001 for s in scores)


def test_exact_sentence_scores_about_one(built):
    text = SENTENCES[4][2]
    top = built.search(text, k=1)[0]
    assert top.sentence == text
    assert top.score == pytest.approx(1.0, abs=1e-4)


def test_k_larger_than_index_returns_everything(built):
    assert len(built.search("detector", k=100)) == len(SENTENCES)


def test_search_many_matches_single_searches(built):
    claims = ["noisy bounding boxes", "soft sampling background"]
    batch = built.search_many(claims, k=2)
    singles = [built.search(c, k=2) for c in claims]
    assert [[h.sentence_id for h in hits] for hits in batch] == [
        [h.sentence_id for h in hits] for hits in singles
    ]


def test_load_from_disk_does_not_re_embed_sentences(index_dir, built):
    model = BagOfWords()
    loaded = EvidenceIndex.load(index_dir, model=model)
    assert model.calls == 0  # nothing embedded just by loading
    claim = "image level labels recover recall"
    assert loaded.search(claim, k=1)[0].sentence_id == built.search(claim, k=1)[0].sentence_id
    assert model.calls == 1  # only the claim


def test_stale_index_is_detected(index_dir, built):
    table = pd.read_csv(index_dir / "sentences.csv")
    table.loc[0, "sentence"] = "A sentence that was edited after the index was built."
    table.to_csv(index_dir / "sentences.csv", index=False)
    with pytest.raises(SystemExit, match="has changed"):
        EvidenceIndex.load(index_dir, model=BagOfWords())


def test_missing_index_gives_a_helpful_message(index_dir):
    with pytest.raises(SystemExit, match="src.index build"):
        EvidenceIndex.load(index_dir, model=BagOfWords())
