"""
Evaluation metrics, split into two families:

1. Retrieval metrics (precision@k, recall@k, MRR, nDCG) -- computed against
   a labeled query set where each query has known "relevant" chunk/doc IDs.
   These are cheap, deterministic, no model required.

2. Generation quality metrics (faithfulness / groundedness, answer
   relevance) -- these need judgment, so they're implemented as an
   LLM-as-judge (see judge.py) with a lexical-overlap fallback that works
   with no model access, clearly labeled as a weaker proxy.

Keeping these separate matters: a RAG system can have perfect retrieval
and still hallucinate at generation time, or have poor retrieval that the
generator "papers over" by refusing to answer. Reporting them together
hides which stage is actually failing.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from rag.retrieval import ScoredChunk


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved: list[ScoredChunk], relevant_doc_ids: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r.chunk.doc_id in relevant_doc_ids)
    return hits / len(top_k)


def recall_at_k(retrieved: list[ScoredChunk], relevant_doc_ids: set[str], k: int) -> float:
    if not relevant_doc_ids:
        return 0.0
    top_k = retrieved[:k]
    hit_docs = {r.chunk.doc_id for r in top_k if r.chunk.doc_id in relevant_doc_ids}
    return len(hit_docs) / len(relevant_doc_ids)


def mrr(retrieved: list[ScoredChunk], relevant_doc_ids: set[str]) -> float:
    for i, r in enumerate(retrieved):
        if r.chunk.doc_id in relevant_doc_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[ScoredChunk], relevant_doc_ids: set[str], k: int) -> float:
    top_k = retrieved[:k]
    dcg = sum(
        (1.0 if r.chunk.doc_id in relevant_doc_ids else 0.0) / math.log2(i + 2)
        for i, r in enumerate(top_k)
    )
    ideal_hits = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(retrieved: list[ScoredChunk], relevant_doc_ids: set[str], k: int) -> float:
    """Average precision at k: mean of precision@i taken at each rank i
    where a relevant doc appears, within the top k. The per-query
    building block for MAP (mean average precision) -- unlike
    precision@k alone, it rewards ranking relevant docs *earlier* among
    the top k, not just including them somewhere in it."""
    top_k = retrieved[:k]
    if not relevant_doc_ids or not top_k:
        return 0.0
    seen_docs: set[str] = set()
    hits = 0
    precisions = []
    for i, r in enumerate(top_k):
        if r.chunk.doc_id in relevant_doc_ids and r.chunk.doc_id not in seen_docs:
            seen_docs.add(r.chunk.doc_id)
            hits += 1
            precisions.append(hits / (i + 1))
    denom = min(len(relevant_doc_ids), k)
    return sum(precisions) / denom if denom > 0 else 0.0


def f1_at_k(retrieved: list[ScoredChunk], relevant_doc_ids: set[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant_doc_ids, k)
    r = recall_at_k(retrieved, relevant_doc_ids, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class RetrievalMetrics:
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    mrr: float
    ndcg_at_k: float
    average_precision: float
    k: int

    def as_dict(self) -> dict:
        return {
            f"precision@{self.k}": self.precision_at_k,
            f"recall@{self.k}": self.recall_at_k,
            f"f1@{self.k}": self.f1_at_k,
            "mrr": self.mrr,
            f"ndcg@{self.k}": self.ndcg_at_k,
            "average_precision": self.average_precision,
        }


def evaluate_retrieval(
    retrieved: list[ScoredChunk], relevant_doc_ids: set[str], k: int = 5
) -> RetrievalMetrics:
    return RetrievalMetrics(
        precision_at_k=precision_at_k(retrieved, relevant_doc_ids, k),
        recall_at_k=recall_at_k(retrieved, relevant_doc_ids, k),
        f1_at_k=f1_at_k(retrieved, relevant_doc_ids, k),
        mrr=mrr(retrieved, relevant_doc_ids),
        ndcg_at_k=ndcg_at_k(retrieved, relevant_doc_ids, k),
        average_precision=average_precision(retrieved, relevant_doc_ids, k),
        k=k,
    )


def mean_average_precision(per_query_ap: list[float]) -> float:
    """MAP: mean of per-query average_precision values. Computed at the
    ablation-runner level (mean across queries for one config), not
    here, since a single query's AP isn't "mean" of anything."""
    return sum(per_query_ap) / len(per_query_ap) if per_query_ap else 0.0


# ---------------------------------------------------------------------------
# Generation metrics (lexical fallback -- see judge.py for LLM-as-judge)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
    "and", "or", "for", "that", "this", "it", "with", "as", "by", "at",
    "be", "has", "have", "had", "not", "i", "don't", "dont",
}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def lexical_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """Cheap groundedness proxy: fraction of the answer's content words
    that also appear somewhere in the retrieved context. Not a substitute
    for LLM-as-judge faithfulness (it can't detect subtle contradiction or
    unsupported causal claims stitched from real words) but useful as a
    fast, free, always-available signal and a sanity check against the
    judge's scores.
    """
    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0
    context_words = set()
    for c in context_chunks:
        context_words |= _content_words(c)
    grounded = answer_words & context_words
    return len(grounded) / len(answer_words)


def refusal_rate(answers: list[str]) -> float:
    """Fraction of answers where the model declined to answer. Tracking
    this matters: a generator that refuses whenever retrieval is weak will
    show artificially high faithfulness (nothing ungrounded is said) while
    being useless. Always report refusal rate alongside faithfulness.
    """
    refusal_markers = ["don't have enough information", "cannot answer",
                        "no information", "not mentioned in the context"]
    refusals = sum(
        1 for a in answers if any(m in a.lower() for m in refusal_markers)
    )
    return refusals / len(answers) if answers else 0.0
