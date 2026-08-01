"""
Ablation runner: sweeps (chunk_strategy x chunk_size x retriever x top_k
x query_transform x reranker) combinations over a labeled eval set,
computes retrieval + generation metrics (and, optionally, LLM-judge
scores and estimated cost) for each, and returns a tidy results table
ready for pandas / plotting. This is the piece that turns "I built a RAG
system" into "I measured what actually moves the needle in a RAG
system."
"""
from __future__ import annotations

import csv
import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path

from rag.chunking import Document
from rag.generation import BaseGenerator
from rag.pipeline import RAGPipeline
from rag.query_transform import QUERY_TRANSFORMS, BaseQueryTransform
from rag.rerank import RERANKERS, BaseReranker
from rag.retrieval import RETRIEVERS
from eval.dataset import EvalExample
from eval.judge import LLMJudge
from eval.metrics import (
    evaluate_retrieval, lexical_faithfulness, refusal_rate, mean_average_precision,
)

# Rough words-per-token constant for a free, offline cost/latency proxy
# (no tokenizer dependency required). Not precise -- it's meant to make
# "which config is cheapest to run" comparable across conditions, not to
# match any specific tokenizer's exact counts.
WORDS_PER_TOKEN = 0.75


def _estimate_tokens(text: str) -> int:
    return max(1, round(len(text.split()) / WORDS_PER_TOKEN))


@dataclass
class AblationConfig:
    chunk_strategies: list[str] = field(default_factory=lambda: ["fixed_size", "recursive", "semantic"])
    chunk_sizes: list[int] = field(default_factory=lambda: [256, 512])
    retriever_names: list[str] = field(default_factory=lambda: ["bm25"])
    top_k_values: list[int] = field(default_factory=lambda: [3, 5])
    query_transform_names: list[str] = field(default_factory=lambda: ["identity"])
    reranker_names: list[str] = field(default_factory=lambda: ["none"])
    # Estimated USD cost per 1000 tokens (prompt + completion combined).
    # Defaults to 0 because the point of this harness is local models;
    # set > 0 to make cost/latency tradeoff curves meaningful if you wire
    # in a paid API generator.
    cost_per_1k_tokens: float = 0.0


def _build_query_transform(name: str, generator: BaseGenerator) -> BaseQueryTransform:
    cls = QUERY_TRANSFORMS[name]
    if name == "identity":
        return cls()
    return cls(generator)


def _build_reranker(name: str) -> BaseReranker:
    return RERANKERS[name]()


def run_ablation(
    documents: list[Document],
    eval_examples: list[EvalExample],
    generator: BaseGenerator,
    config: AblationConfig,
    run_generation: bool = True,
    judge: LLMJudge | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Runs the full grid and returns a list of result dicts, one row per
    (config, query) pair -- suitable for `pd.DataFrame(results)`.

    If `judge` is provided (an eval.judge.LLMJudge wrapping any
    BaseGenerator), each row also gets LLM-judge faithfulness/relevance
    scores alongside the lexical-overlap fallback -- pass one explicitly
    so the ablation runner never silently spins up a model on its own.
    """
    results = []
    combos = list(itertools.product(
        config.chunk_strategies, config.chunk_sizes,
        config.retriever_names, config.top_k_values,
        config.query_transform_names, config.reranker_names,
    ))

    for chunk_strategy, chunk_size, retriever_name, top_k, transform_name, reranker_name in combos:
        chunk_kwargs = {} if chunk_strategy == "semantic" else {"chunk_size": chunk_size}
        retriever_cls = RETRIEVERS[retriever_name]
        retriever = retriever_cls()
        query_transform = _build_query_transform(transform_name, generator)
        reranker = _build_reranker(reranker_name)

        pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            chunk_strategy=chunk_strategy,
            chunk_kwargs=chunk_kwargs,
            top_k=top_k,
            query_transform=query_transform,
            reranker=reranker,
        )

        t0 = time.time()
        pipeline.ingest(documents)
        index_time = time.time() - t0

        if verbose:
            print(f"[{chunk_strategy} size={chunk_size} | {retriever_name} | k={top_k} | "
                  f"transform={transform_name} | rerank={reranker_name}] "
                  f"{len(pipeline.chunks)} chunks indexed in {index_time:.2f}s")

        answers_for_refusal = []
        per_query_ap = []
        for ex in eval_examples:
            rag_result = pipeline.query(ex.question)

            retrieval_m = evaluate_retrieval(
                rag_result.retrieved_chunks, set(ex.relevant_doc_ids), k=top_k
            )
            per_query_ap.append(retrieval_m.average_precision)

            row = {
                "query_id": ex.query_id,
                "chunk_strategy": chunk_strategy,
                "chunk_size": chunk_size if chunk_strategy != "semantic" else None,
                "retriever": retriever_name,
                "top_k": top_k,
                "query_transform": transform_name,
                "reranker": reranker_name,
                "num_chunks_indexed": len(pipeline.chunks),
                "index_time_sec": round(index_time, 3),
                "retrieval_latency_sec": round(rag_result.retrieval_latency_sec, 4),
                "generation_latency_sec": round(rag_result.generation_latency_sec, 4),
                **retrieval_m.as_dict(),
            }

            if run_generation:
                context_texts = [r.chunk.text for r in rag_result.retrieved_chunks]
                row["lexical_faithfulness"] = lexical_faithfulness(
                    rag_result.answer, context_texts
                )
                row["answer"] = rag_result.answer
                answers_for_refusal.append(rag_result.answer)

                prompt_tokens = _estimate_tokens(ex.question + " ".join(context_texts))
                completion_tokens = _estimate_tokens(rag_result.answer)
                total_tokens = prompt_tokens + completion_tokens
                row["estimated_tokens"] = total_tokens
                row["estimated_cost_usd"] = round(
                    total_tokens / 1000 * config.cost_per_1k_tokens, 6
                )

                if judge is not None:
                    faith = judge.score_faithfulness(ex.question, rag_result.answer, context_texts)
                    rel = judge.score_relevance(ex.question, rag_result.answer)
                    row["judge_faithfulness"] = faith.score
                    row["judge_faithfulness_reason"] = faith.reason
                    row["judge_relevance"] = rel.score
                    row["judge_relevance_reason"] = rel.reason

            results.append(row)

        if run_generation and answers_for_refusal:
            r_rate = refusal_rate(answers_for_refusal)
            m_ap = mean_average_precision(per_query_ap)
            for row in results[-len(eval_examples):]:
                row["refusal_rate_for_config"] = round(r_rate, 3)
                row["map_for_config"] = round(m_ap, 4)

    return results


def save_results_csv(results: list[dict], path: str | Path) -> None:
    if not results:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in results for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
