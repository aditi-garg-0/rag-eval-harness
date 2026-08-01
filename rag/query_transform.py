"""
Query transforms: interventions applied *before* first-stage retrieval,
on the theory that the user's literal question is often a worse search
query than something derived from it. Three well-known techniques from
the retrieval literature, each with a stated failure mode it targets:

- HyDE (Gao et al., 2022): the question and the passage that answers it
  often don't share vocabulary ("why did X fail?" vs. a passage
  describing the failure without using the word "fail"). Asking the
  generator to sketch a hypothetical answer first, then retrieving with
  *that* as the query, closes some of that lexical gap.

- Multi-query fusion (RAG-Fusion style): a single phrasing of a question
  may miss a chunk that uses different terminology. Generating a handful
  of varied search queries and fusing their retrieved candidates via
  reciprocal rank fusion (RRF) hedges against any one phrasing missing.

- Query decomposition (multi-hop): compound questions ("how does X
  compare to Y on Z?") need evidence that may live in more than one
  document. Single-shot retrieval on the whole question tends to
  over-fit to whichever sub-topic is most prominent lexically. Breaking
  it into atomic sub-questions and retrieving per-hop, then merging,
  gives each sub-topic a fair shot at being retrieved.

All three need a real generator to do the actual rewriting -- with
MockGenerator (no real language understanding) they degrade to plain
identity retrieval rather than doing something meaningless with mock
text, so ablation runs stay well-defined (not silently broken) offline.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from rag.generation import BaseGenerator, MockGenerator
from rag.retrieval import BaseRetriever, ScoredChunk

HYDE_PROMPT = """Write a short hypothetical passage (2-4 sentences) that would answer \
the following question, as if it were an excerpt from a reference document. Do not \
say you don't know -- just write your best-guess passage, even if uncertain.

Question: {question}

Hypothetical passage:"""

MULTI_QUERY_PROMPT = """Generate {n} different search queries that could each retrieve \
documents relevant to answering the question below. Vary phrasing and emphasis (e.g. \
synonyms, more specific sub-aspects, more general framing). One query per line, no \
numbering, no other text.

Question: {question}

Search queries:"""

DECOMPOSE_PROMPT = """Break the following question into 2-4 simpler, atomic \
sub-questions whose answers together would answer the original question. If the \
question is already atomic and cannot be meaningfully decomposed, output just the \
original question. One sub-question per line, no numbering, no other text.

Question: {question}

Sub-questions:"""


def _rrf_fuse(
    ranked_lists: list[list[ScoredChunk]], top_k: int, rrf_k: int = 60
) -> list[ScoredChunk]:
    """Reciprocal rank fusion: combine several independently-ranked
    candidate lists into one, using each item's *rank* (not its raw
    score, which isn't comparable across queries/retrievers) via
    1 / (rrf_k + rank). Standard, parameter-light fusion method used in
    RAG-Fusion and classic multi-query IR.
    """
    fused_scores: dict[str, float] = {}
    chunk_by_id = {}
    for ranked in ranked_lists:
        for rank, sc in enumerate(ranked):
            cid = sc.chunk.chunk_id
            chunk_by_id[cid] = sc.chunk
            fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
    ordered = sorted(fused_scores.items(), key=lambda x: -x[1])[:top_k]
    return [ScoredChunk(chunk=chunk_by_id[cid], score=score) for cid, score in ordered]


class BaseQueryTransform(ABC):
    name: str = "identity"

    @abstractmethod
    def get_chunks(
        self, question: str, retriever: BaseRetriever, top_k: int
    ) -> list[ScoredChunk]:
        ...


class IdentityTransform(BaseQueryTransform):
    """The "no transform" baseline condition: retrieve on the question
    exactly as asked. Every other transform in this module is measured
    against this."""

    name = "identity"

    def get_chunks(self, question: str, retriever: BaseRetriever, top_k: int) -> list[ScoredChunk]:
        return retriever.retrieve(question, top_k=top_k)


class HyDETransform(BaseQueryTransform):
    name = "hyde"

    def __init__(self, generator: BaseGenerator):
        self.generator = generator

    def get_chunks(self, question: str, retriever: BaseRetriever, top_k: int) -> list[ScoredChunk]:
        if isinstance(self.generator, MockGenerator):
            # A mock generator can't produce a meaningful hypothetical
            # passage -- fall back to identity rather than retrieving
            # against the literal string "[MOCK GENERATION ...]".
            return retriever.retrieve(question, top_k=top_k)
        hypothetical = self.generator.generate(HYDE_PROMPT.format(question=question))
        if not hypothetical.strip():
            return retriever.retrieve(question, top_k=top_k)
        return retriever.retrieve(hypothetical, top_k=top_k)


class MultiQueryFusionTransform(BaseQueryTransform):
    name = "multi_query"

    def __init__(self, generator: BaseGenerator, n_queries: int = 3, rrf_k: int = 60):
        self.generator = generator
        self.n_queries = n_queries
        self.rrf_k = rrf_k

    def _fallback_queries(self, question: str) -> list[str]:
        """Deterministic, model-free query variants so this transform is
        still exercisable in offline tests / --quick runs: the question
        as-is, a stopword-stripped keyword version, and (if long enough)
        its first half / second half as separate queries."""
        stopwords = {"the", "a", "an", "is", "are", "of", "to", "in", "on", "and",
                     "or", "for", "what", "how", "why", "does", "do", "did"}
        words = re.findall(r"[a-zA-Z0-9]+", question.lower())
        keywords = " ".join(w for w in words if w not in stopwords)
        variants = [question, keywords or question]
        if len(words) >= 6:
            mid = len(words) // 2
            variants.append(" ".join(words[:mid]))
            variants.append(" ".join(words[mid:]))
        return variants[: self.n_queries] or [question]

    def get_chunks(self, question: str, retriever: BaseRetriever, top_k: int) -> list[ScoredChunk]:
        if isinstance(self.generator, MockGenerator):
            queries = self._fallback_queries(question)
        else:
            raw = self.generator.generate(
                MULTI_QUERY_PROMPT.format(question=question, n=self.n_queries)
            )
            queries = [q.strip() for q in raw.splitlines() if q.strip()][: self.n_queries]
            if not queries:
                queries = [question]

        fetch_k = max(top_k * 2, 10)
        ranked_lists = [retriever.retrieve(q, top_k=fetch_k) for q in queries]
        return _rrf_fuse(ranked_lists, top_k=top_k, rrf_k=self.rrf_k)


class DecomposeTransform(BaseQueryTransform):
    """Multi-hop decomposition. Splits the question into sub-questions,
    retrieves separately per sub-question, then merges the union by best
    per-chunk score across hops (a chunk that answers any hop well is
    kept; the point is coverage across sub-topics, not agreement)."""

    name = "decompose"

    def __init__(self, generator: BaseGenerator, max_subquestions: int = 4):
        self.generator = generator
        self.max_subquestions = max_subquestions

    def _fallback_subquestions(self, question: str) -> list[str]:
        """Offline-safe decomposition: split on coordinating conjunctions
        ("and", "vs", "compared to") if present, else treat the question
        as already atomic. Much cruder than an LLM decomposition, but
        lets this transform's *merging* logic be exercised without a
        real model."""
        parts = re.split(r"\s+(?:and|vs\.?|versus|compared to)\s+", question, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        return parts if len(parts) > 1 else [question]

    def get_chunks(self, question: str, retriever: BaseRetriever, top_k: int) -> list[ScoredChunk]:
        if isinstance(self.generator, MockGenerator):
            subquestions = self._fallback_subquestions(question)
        else:
            raw = self.generator.generate(DECOMPOSE_PROMPT.format(question=question))
            subquestions = [q.strip() for q in raw.splitlines() if q.strip()][: self.max_subquestions]
            if not subquestions:
                subquestions = [question]

        best: dict[str, ScoredChunk] = {}
        for sub_q in subquestions:
            for sc in retriever.retrieve(sub_q, top_k=top_k):
                cid = sc.chunk.chunk_id
                if cid not in best or sc.score > best[cid].score:
                    best[cid] = sc
        ranked = sorted(best.values(), key=lambda sc: -sc.score)[:top_k]
        return ranked


QUERY_TRANSFORMS: dict[str, type[BaseQueryTransform]] = {
    "identity": IdentityTransform,
    "hyde": HyDETransform,
    "multi_query": MultiQueryFusionTransform,
    "decompose": DecomposeTransform,
}
