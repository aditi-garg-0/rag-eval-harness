"""
Reranking stage: an optional second pass over the retriever's candidate
set, scored with a model that looks at the (query, chunk) pair jointly
rather than via independent embeddings/term stats. This is the standard
way production RAG stacks close the gap between "fast first-stage
retrieval" (BM25/dense, which score query and document independently)
and "expensive but accurate" relevance judgment.

Kept as its own ablation axis (see eval/ablation.py) rather than baked
into a retriever, because the interesting empirical question is "how
much does reranking help *on top of* each retriever", not "which single
retriever+reranker combo wins" -- those are different questions and
conflating them is exactly the kind of thing this project's README
argues against.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from rag.retrieval import ScoredChunk


class BaseReranker(ABC):
    name: str = "base"

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        """Takes a (typically larger-than-top_k) candidate set from a
        first-stage retriever and returns the best `top_k`, re-scored."""
        ...


class NoOpReranker(BaseReranker):
    """The "none" condition in the ablation grid: whatever the retriever
    returned, unchanged. This is the baseline every other reranker is
    measured against -- without it, "reranking helps" is unfalsifiable."""

    name = "none"

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return candidates[:top_k]


class CrossEncoderReranker(BaseReranker):
    """Scores each (query, chunk) pair jointly with a cross-encoder
    (default: ms-marco-MiniLM, a standard small reranking model). This is
    the "real" reranker -- meaningfully more accurate than first-stage
    retrieval because the model can attend across query and document
    tokens together, at the cost of one forward pass per candidate
    (doesn't scale to scoring the whole corpus, hence "rerank the
    retriever's top candidates" rather than "replace the retriever").

    Requires internet access to download the model on first use (same
    lazy-import pattern as DenseRetriever, so importing this module never
    requires the dependency).
    """

    name = "cross_encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers not installed (needed for "
                    "CrossEncoderReranker). Run: pip install sentence-transformers"
                ) from e
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = model.predict(pairs)
        rescored = [
            ScoredChunk(chunk=c.chunk, score=float(s))
            for c, s in zip(candidates, scores)
        ]
        rescored.sort(key=lambda sc: -sc.score)
        return rescored[:top_k]


class LexicalOverlapReranker(BaseReranker):
    """Offline, dependency-free fallback and baseline reranker. Rescales
    candidates by exact query-term overlap (Jaccard over token sets)
    rather than BM25's IDF/length-normalized score. Not a substitute for
    a real cross-encoder -- it can't detect paraphrase or semantic
    relevance, only exact lexical overlap the first-stage retriever may
    have already used -- but it keeps "reranking" testable as an
    ablation axis with zero downloads, and gives a sanity-check baseline
    for how much of a cross-encoder's lift is "real" versus something
    this much simpler already captures.
    """

    name = "lexical_overlap"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(text.lower().split())

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        q_tokens = self._tokens(query)
        rescored = []
        for c in candidates:
            c_tokens = self._tokens(c.chunk.text)
            union = q_tokens | c_tokens
            overlap = len(q_tokens & c_tokens) / len(union) if union else 0.0
            rescored.append(ScoredChunk(chunk=c.chunk, score=overlap))
        rescored.sort(key=lambda sc: -sc.score)
        return rescored[:top_k]


RERANKERS: dict[str, type[BaseReranker]] = {
    "none": NoOpReranker,
    "lexical_overlap": LexicalOverlapReranker,
    "cross_encoder": CrossEncoderReranker,
}
