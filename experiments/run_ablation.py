"""
Runs the full ablation sweep and writes results to experiments/results/.

Quick start (fully offline, no model download, uses BM25 + MockGenerator):
    python experiments/run_ablation.py --quick

Real run (needs `ollama serve` running locally with a pulled model):
    python experiments/run_ablation.py --generator ollama --ollama-model llama3.2 \
        --retrievers bm25 hybrid --corpus data/corpus.json

Advanced run exercising every axis (reranking, query transforms, LLM judge):
    python experiments/run_ablation.py --generator ollama --ollama-model llama3.2 \
        --retrievers bm25 hybrid --query-transforms identity hyde multi_query decompose \
        --rerankers none lexical_overlap cross_encoder --use-llm-judge
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.corpus import load_corpus
from eval.ablation import AblationConfig, run_ablation, save_results_csv
from eval.dataset import load_eval_set
from eval.judge import LLMJudge
from rag.generation import GENERATORS, MockGenerator
from rag.query_transform import QUERY_TRANSFORMS
from rag.rerank import RERANKERS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="data/sample_corpus/sample_corpus.json")
    parser.add_argument("--eval-set", default="data/sample_corpus/sample_eval_set.json")
    parser.add_argument("--generator", choices=list(GENERATORS), default="mock")
    parser.add_argument("--ollama-model", default="llama3.2")
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--retrievers", nargs="+", default=["bm25"],
                         choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--chunk-strategies", nargs="+",
                         default=["fixed_size", "recursive", "semantic"])
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--top-k", nargs="+", type=int, default=[3, 5])
    parser.add_argument("--query-transforms", nargs="+", default=["identity"],
                         choices=list(QUERY_TRANSFORMS),
                         help="Query rewriting strategies to sweep (identity/hyde/"
                              "multi_query/decompose). Non-identity transforms need a "
                              "real generator to do anything beyond a deterministic "
                              "fallback -- see rag/query_transform.py.")
    parser.add_argument("--rerankers", nargs="+", default=["none"],
                         choices=list(RERANKERS),
                         help="Second-stage reranking to sweep (none/lexical_overlap/"
                              "cross_encoder). cross_encoder downloads a model on first use.")
    parser.add_argument("--use-llm-judge", action="store_true",
                         help="Score faithfulness/relevance with an LLM judge (using the "
                              "same generator) in addition to the lexical fallback. "
                              "Meaningfully slower -- two extra generation calls per query "
                              "per config.")
    parser.add_argument("--cost-per-1k-tokens", type=float, default=0.0,
                         help="USD per 1000 tokens, for the estimated_cost_usd column. "
                              "0 (default) is correct for local models; set >0 if you've "
                              "wired in a paid API generator.")
    parser.add_argument("--out", default="experiments/results/ablation_results.csv")
    parser.add_argument("--quick", action="store_true",
                         help="Fast offline smoke test: bm25 only, mock generator, one chunk size")
    args = parser.parse_args()

    if args.quick:
        args.retrievers = ["bm25"]
        args.chunk_sizes = [256]
        args.top_k = [3]
        args.generator = "mock"
        args.query_transforms = ["identity"]
        args.rerankers = ["none"]
        args.use_llm_judge = False

    documents = load_corpus(args.corpus)
    eval_examples = load_eval_set(args.eval_set)
    print(f"Loaded {len(documents)} documents, {len(eval_examples)} eval queries.")

    if args.generator == "ollama":
        generator = GENERATORS["ollama"](model=args.ollama_model)
    elif args.generator == "huggingface":
        generator = GENERATORS["huggingface"](model_name=args.hf_model)
    else:
        generator = MockGenerator()
        print("Using MockGenerator: generation-quality metrics will not reflect a "
              "real model, and hyde/multi_query/decompose query transforms fall back "
              "to deterministic non-model behavior (see rag/query_transform.py). This "
              "mode exists to sanity-check the pipeline and retrieval metrics only.")

    if args.use_llm_judge and isinstance(generator, MockGenerator):
        print("--use-llm-judge with a mock generator would just parse-fail on "
              "'[MOCK GENERATION ...]' -- ignoring the flag.")
        args.use_llm_judge = False

    judge = LLMJudge(generator) if args.use_llm_judge else None

    config = AblationConfig(
        chunk_strategies=args.chunk_strategies,
        chunk_sizes=args.chunk_sizes,
        retriever_names=args.retrievers,
        top_k_values=args.top_k,
        query_transform_names=args.query_transforms,
        reranker_names=args.rerankers,
        cost_per_1k_tokens=args.cost_per_1k_tokens,
    )

    results = run_ablation(documents, eval_examples, generator, config, judge=judge)
    save_results_csv(results, args.out)
    print(f"\nSaved {len(results)} result rows to {args.out}")

    # Quick console summary
    from collections import defaultdict
    agg = defaultdict(list)
    for row in results:
        key = (row["chunk_strategy"], row["chunk_size"], row["retriever"], row["top_k"],
               row["query_transform"], row["reranker"])
        agg[key].append(row)

    print("\n--- Summary (avg retrieval metrics per config) ---")
    for key, rows in sorted(agg.items(), key=lambda x: -sum(r.get(f"precision@{x[0][3]}", 0) for r in x[1])):
        strat, size, retr, k, transform, rerank = key
        p = sum(r.get(f"precision@{k}", 0) for r in rows) / len(rows)
        r = sum(r.get(f"recall@{k}", 0) for r in rows) / len(rows)
        m = sum(r.get("mrr", 0) for r in rows) / len(rows)
        map_score = rows[0].get("map_for_config", 0.0)
        print(f"  {strat:12s} size={str(size):5s} {retr:6s} k={k}  "
              f"xform={transform:10s} rerank={rerank:15s}  "
              f"precision@{k}={p:.3f}  recall@{k}={r:.3f}  mrr={m:.3f}  map={map_score:.3f}")

    print("\nTip: use eval/significance.py's compare_configs() to test whether any two "
          "of the configs above differ significantly, rather than eyeballing means on "
          f"an {len(eval_examples)}-query eval set.")


if __name__ == "__main__":
    main()
