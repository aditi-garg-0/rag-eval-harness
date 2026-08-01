# RAG Evaluation Harness — Findings

*Fill this in as you run experiments. Treat it like a mini research report: state a hypothesis, show the data, state what you actually found (including when it contradicts the hypothesis).*

## 1. Setup

- **Corpus**: [e.g. N papers fetched from arXiv on topics: retrieval-augmented generation, dense retrieval, hallucination detection]
- **Eval set**: N labeled queries (M single-hop, M multi-hop, M adversarial/no-answer)
- **Retriever(s) tested**: BM25 / Dense (`model name`) / Hybrid (α = ...)
- **Generator**: e.g. Ollama running `llama3.2:3b`, or `Qwen2.5-1.5B-Instruct` via HF
- **Judge**: same/different model as generator? (state explicitly — using the same model as both generator and judge is a known confound worth flagging)

## 2. Hypotheses going in

State 2-4 things you expected before running the sweep. Example:

1. Smaller chunks → better retrieval precision, worse generation quality (less context).
2. Hybrid retrieval beats BM25 alone on paraphrased/multi-hop queries specifically, not uniformly.
3. Faithfulness will correlate with retrieval precision, but not perfectly — some queries will get faithful answers built on marginally-relevant context (the model "fills in" reasonably).
4. The adversarial no-answer query will reveal whether the generator has a real refusal mechanism or just always answers.

## 3. Results

### 3.1 Chunking strategy × chunk size (retrieval metrics)

*Paste/summarize the table from `experiments/results/ablation_results.csv`, grouped by config. A markdown table or a chart (bar chart of precision/recall per config) works well here.*

| chunk_strategy | chunk_size | retriever | top_k | precision@k | recall@k | MRR |
|---|---|---|---|---|---|---|
| | | | | | | |

**Observation:** ...

### 3.2 Retriever comparison (BM25 vs dense vs hybrid)

**Observation:** ...

### 3.3 Generation quality: faithfulness & refusal rate

Report both lexical-overlap faithfulness (`eval/metrics.py`) and LLM-judge faithfulness (`eval/judge.py`) side by side — do they agree? Where do they diverge, and on what kind of question?

**Observation:** ...

### 3.4 The adversarial query (no relevant document exists)

What did the system do with `q8` (or your own no-answer query)? Refuse cleanly, hallucinate from parametric memory, or something in between? This one data point is disproportionately informative about real-world reliability.

**Observation:** ...

## 4. Judge calibration check

Take 10-15 (question, answer, judge score) triples and score them yourself. Report agreement (exact match, or within 1 point) between you and the judge. If they disagree a lot, say so — that's a real finding, not a failure of the project. Small local models as judges are known to be noisier than GPT-4-class judges; quantifying that gap for *your* setup is more valuable than assuming it away.

| Query | Judge score | Your score | Agree? |
|---|---|---|---|
| | | | |

## 5. What actually moved the needle

Rank the factors you tested (chunking strategy, chunk size, retriever type, top_k, reranking) by how much they affected final answer quality, based on your data — not intuition. Which ones mattered less than expected?

## 6. Limitations

- Corpus size / eval set size (be honest about statistical power — 8-30 queries is directional, not conclusive)
- Judge model limitations
- Anything you didn't get to test (reranking, larger models, larger corpus)

## 7. What I'd do with more time/compute

...
