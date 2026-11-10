"""The CiteAssist web app.

Run from the project root:
    streamlit run app.py

Paste a paragraph, press "Find evidence", and for each claim the app shows
the evidence sentences as quotes with a label (supports / contradicts /
related), the LLM's reason, and an APA citation, followed by a reference list.
"""

from __future__ import annotations

from collections import Counter

import streamlit as st

from src.cite import load_papers, reference
from src.claims import split_claims
from src.index import EvidenceIndex
from src.judge import Judge, LLMError
from src.pipeline import CheckedClaim, CiteAssist, papers_used, report_markdown
from src.retrieve import THRESHOLD, Retriever, load_reranker

MAX_CLAIMS = 15  # keeps one run within free-tier LLM rate limits

EXAMPLE = (
    "Detectors such as Faster R-CNN remain fairly robust when many annotations are missing. "
    "Noisy bounding box locations degrade the accuracy of object detectors. "
    "Correcting noisy annotations before training improves detection performance. "
    "Coffee improves sleep quality."
)

LABEL_BADGE = {
    "supports": ":green-badge[Supports]",
    "contradicts": ":red-badge[Contradicts]",
    "related": ":gray-badge[Related]",
    None: ":blue-badge[Evidence]",
}
VERDICT_BADGE = {
    "supported": ":green-badge[:material/check_circle: Supported]",
    "contradicted": ":red-badge[:material/cancel: Contradicted]",
    "contested": ":orange-badge[:material/balance: Contested]",
    "related": ":gray-badge[:material/info: Only related evidence]",
    "found": ":blue-badge[:material/search: Evidence found]",
    "no evidence": ":violet-badge[:material/help: No evidence found]",
}

st.set_page_config(page_title="CiteAssist", page_icon=":material/menu_book:", layout="centered")


# ---------------------------------------------------------------- cached resources
# Loaded once per server and shared by every run, so only the first search is slow.


@st.cache_resource(show_spinner="Loading the search index and embedding model...")
def get_index() -> EvidenceIndex:
    return EvidenceIndex.load()


@st.cache_resource(show_spinner="Loading the re-ranking model...")
def get_reranker():
    return load_reranker()


@st.cache_resource(show_spinner=False)
def get_judge() -> tuple[Judge | None, str | None]:
    """The LLM judge, or (None, reason) if it isn't set up."""
    try:
        return Judge.from_env(), None
    except SystemExit as exc:  # provider_from_env exits with a setup message
        return None, str(exc)


@st.cache_data(show_spinner=False)
def get_papers():
    return load_papers()


def load_or_stop(loader, what: str):
    try:
        return loader()
    except SystemExit as exc:  # our loaders exit with a helpful message
        st.error(f"Could not load the {what}: {exc}", icon=":material/error:")
        st.stop()


# ---------------------------------------------------------------- sidebar


def sidebar(index: EvidenceIndex, papers: dict) -> tuple[bool, bool, float]:
    with st.sidebar:
        st.header("Settings")
        use_judge = st.toggle("Label evidence with the LLM", value=True,
                              help="Turn off to see evidence and citations without any API calls.")
        rerank = st.toggle("Re-rank with the cross-encoder", value=True,
                           help="Reads each claim and sentence together to order the evidence more precisely.")
        threshold = st.slider("Minimum similarity", 0.20, 0.70, THRESHOLD, 0.05,
                              help="Sentences less similar than this are ignored. Higher = stricter; "
                                   "claims with nothing above it show 'No evidence found'.")

        judge, problem = get_judge()
        if use_judge and judge is not None:
            st.caption(f"LLM: {judge.provider.name} · `{judge.provider.model}`")
        elif use_judge:
            st.warning(f"LLM not set up, so evidence is shown without labels. {problem}",
                       icon=":material/key_off:")

        st.divider()
        counts = index.table["paper_id"].value_counts()
        st.header(f"Indexed papers ({len(counts)})")
        st.caption(f"{len(index):,} sentences in the index")
        for paper_id in sorted(counts.index):
            paper = papers.get(paper_id)
            if paper is None:
                st.markdown(f"**{paper_id}** · not in papers.csv")
                continue
            st.markdown(f"**{paper.title}**  \n"
                        f"{paper.authors[0][0]}{' et al.' if len(paper.authors) > 2 else ''}, "
                        f"{paper.year} · {counts[paper_id]} sentences")
    return use_judge, rerank, threshold


# ---------------------------------------------------------------- results


def summary(checked: list[CheckedClaim]) -> None:
    verdicts = Counter(c.verdict for c in checked)
    cols = st.columns(4)
    cols[0].metric("Claims", len(checked))
    cols[1].metric("Supported", verdicts["supported"])
    cols[2].metric("Contradicted / contested", verdicts["contradicted"] + verdicts["contested"])
    cols[3].metric("No evidence", verdicts["no evidence"])


def claim_card(c: CheckedClaim) -> None:
    with st.container(border=True):
        st.markdown(f"**Claim {c.claim.index}** &nbsp; {VERDICT_BADGE[c.verdict]}")
        st.markdown(f"*{c.claim.text}*")
        if not c.found:
            st.info("No sentence in the indexed papers is close enough to this claim. "
                    "It may need a source outside these papers, or a lower minimum similarity.",
                    icon=":material/search_off:")
            return
        for item in c.items:
            with st.container(border=True):
                st.markdown(f"{LABEL_BADGE[item.label]} &nbsp; **{item.citation}**")
                st.markdown(f"> “{item.evidence.sentence}”")
                if item.judgement:
                    fallback = " (the LLM's answer was unusable)" if item.judgement.source == "fallback" else ""
                    st.caption(f"**Why:** {item.judgement.reason}{fallback}")
                scores = f"similarity {item.evidence.similarity:.2f}"
                if item.evidence.rerank_score is not None:
                    scores += f" · re-rank {item.evidence.rerank_score:.2f}"
                st.caption(f"{item.paper.title} · {scores}")


def references(checked: list[CheckedClaim]) -> None:
    refs = papers_used(checked)
    if not refs:
        return
    st.subheader("References")
    for paper in refs:
        st.markdown(reference(paper, markdown=True))
    with st.expander("Copy as plain text"):
        st.code("\n\n".join(reference(p) for p in refs), language=None, wrap_lines=True)


# ---------------------------------------------------------------- page


def main() -> None:
    st.title(":material/menu_book: CiteAssist")
    st.write("Paste a paragraph you've written. For each claim, CiteAssist finds sentences in the "
             "indexed papers that support or contradict it, and gives you a ready-made APA citation. "
             "Every quote is a real sentence from a real paper, with its page.")

    index = load_or_stop(get_index, "search index")
    papers = load_or_stop(get_papers, "paper list (data/papers.csv)")
    use_judge, rerank, threshold = sidebar(index, papers)

    if "paragraph" not in st.session_state:
        st.session_state.paragraph = ""

    def fill_example():
        st.session_state.paragraph = EXAMPLE

    st.text_area("Your paragraph", key="paragraph", height=180,
                 placeholder="e.g. Label noise reduces the accuracy of object detection models. ...")
    left, right = st.columns([1, 1])
    run = left.button("Find evidence", type="primary", icon=":material/search:", width="stretch")
    right.button("Try an example", on_click=fill_example, width="stretch")

    if run:
        paragraph = st.session_state.paragraph.strip()
        if not paragraph:
            st.warning("Paste a paragraph first.", icon=":material/edit_note:")
            return
        reranker = get_reranker() if rerank else None
        judge = get_judge()[0] if use_judge else None
        retriever = Retriever(index, reranker, threshold=threshold)
        assistant = CiteAssist(retriever, judge, papers)

        claims = split_claims(paragraph)
        if not claims:
            st.warning("No claims found. Each sentence needs at least 4 words.", icon=":material/edit_note:")
            return
        if len(claims) > MAX_CLAIMS:
            st.warning(f"Checking the first {MAX_CLAIMS} of {len(claims)} claims, to stay within "
                       "the LLM's rate limits.", icon=":material/speed:")
            claims = claims[:MAX_CLAIMS]

        with st.spinner(f"Checking {len(claims)} claim{'s' if len(claims) != 1 else ''}..."):
            try:
                checked = assistant.check_claims(claims)
            except LLMError as exc:
                st.warning(f"The LLM could not be reached ({exc}). Showing evidence without labels.",
                           icon=":material/cloud_off:")
                assistant.judge = None
                checked = assistant.check_claims(claims)
        st.session_state.results = checked

    checked = st.session_state.get("results")
    if not checked:
        return
    st.divider()
    summary(checked)
    for c in checked:
        claim_card(c)
    references(checked)
    st.download_button("Download results (.md)", report_markdown(checked), file_name="citeassist_results.md",
                       mime="text/markdown", icon=":material/download:")


main()
