# RAG Evaluation Harness

A retrieval-augmented generation system built around a central question:
**where does a RAG pipeline actually fail, and which design choices matter?**

Most RAG repos stop at "it retrieves, it generates, it works on my demo
query." This one is built the other way around: the pipeline exists so it
can be systematically ablated. Chunking strategy, chunk size, retriever
type (sparse/dense/hybrid), and top-k are all pluggable and swept as
independent axes, with retrieval and generation quality measured
*separately* — because a RAG system can have great retrieval and still
hallucinate, or bad retrieval that the generator quietly refuses around,
and averaging those together hides which stage is the actual bottleneck.

Runs fully offline/local by default (BM25 + a deterministic mock
generator for pipeline testing), with pluggable dense retrieval and local
LLM generation (via [Ollama](https://ollama.com) or HuggingFace
`transformers`) for real experiments.

## Why this exists

Most portfolio RAG projects show that something *works*. This one is
designed to show *why* — an evaluation harness with real ablations,
adversarial test cases (a query with no answer in the corpus, to check
whether the system hallucinates instead of refusing), and an honest
findings write-up rather than a cherry-picked demo. See
[`reports/findings.md`](reports/findings.md) for a template and
[the design writeup below](#design-notes) for the reasoning behind each
piece.

## Architecture

```
data/
  fetch_corpus.py      # pulls papers from the arXiv API (needs internet)
  corpus.py             # loads corpus.json -> Document objects
  sample_corpus/         # small hand-built offline corpus + labeled eval set
rag/
  chunking.py            # fixed_size / recursive / semantic chunkers
  retrieval.py           # BM25 (sparse), dense (sentence-transformers), hybrid
  query_transform.py     # HyDE, multi-query fusion (RAG-Fusion), multi-hop decomposition
  rerank.py               # cross-encoder reranking (+ offline lexical-overlap fallback)
  generation.py          # Ollama / HuggingFace / Mock generator backends
  pipeline.py            # ties chunking -> query transform -> retrieval -> rerank -> generation
eval/
  metrics.py              # precision@k, recall@k, F1@k, MRR, nDCG, MAP, lexical faithfulness, refusal rate
  judge.py                 # LLM-as-judge for faithfulness & relevance (auditable JSON scores)
  judge_calibration.py       # agreement (weighted kappa, MAE) between judge scores and human labels
  significance.py             # bootstrap CIs + paired permutation tests for comparing configs
  dataset.py                # labeled EvalExample loader
  ablation.py                # sweeps configs (chunking x retrieval x transform x rerank), tidy results table
experiments/
  run_ablation.py             # CLI entry point
  results/                     # CSV output lands here
reports/
  findings.md                  # research write-up template
tests/                          # offline unit tests (chunking, BM25, metrics, rerank, transforms, stats)
```

## Quick start (fully offline, ~5 seconds, no downloads)

```bash
pip install -r requirements.txt
python experiments/run_ablation.py --quick
```

This runs the bundled 8-document sample corpus (about RAG concepts
themselves — chunking, BM25, dense retrieval, hallucination, etc.)
against an 8-query labeled eval set (including one adversarial
no-relevant-document query) using BM25 retrieval and a mock generator.
It's meant to prove the plumbing works end-to-end, not to produce real
findings — swap in a real generator for that (see below).

## Running with a real local model

Install [Ollama](https://ollama.com), pull a small model, and run:

```bash
ollama pull llama3.2
ollama serve  # in a separate terminal

python experiments/run_ablation.py \
  --generator ollama --ollama-model llama3.2 \
  --retrievers bm25 \
  --chunk-strategies fixed_size recursive semantic \
  --chunk-sizes 256 512 \
  --top-k 3 5
```

Or with HuggingFace `transformers` directly (needs `pip install
transformers torch` and a model download):

```bash
python experiments/run_ablation.py --generator huggingface \
  --hf-model Qwen/Qwen2.5-1.5B-Instruct
```

## Building your own corpus

The bundled sample corpus is tiny (8 docs) by design, for fast offline
testing. For a real ablation study, fetch a real corpus from arXiv:

```bash
python data/fetch_corpus.py \
  --queries "retrieval augmented generation" "dense passage retrieval" \
             "hallucination detection LLM" "in-context learning" \
  --max-per-query 40 --out data/corpus.json
```

Then write a labeled eval set matching `data/sample_corpus/sample_eval_set.json`'s
format — a handful of questions with the doc_ids that actually answer
them (include at least one adversarial "no correct answer exists in the
corpus" query; it's one of the most informative test cases for measuring
whether your generator hallucinates or refuses appropriately).

## Enabling dense/hybrid retrieval

```bash
pip install sentence-transformers
python experiments/run_ablation.py --generator ollama --ollama-model llama3.2 \
  --retrievers bm25 dense hybrid
```

`DenseRetriever` downloads `sentence-transformers/all-MiniLM-L6-v2` from
HuggingFace on first use (swap the model name in `rag/retrieval.py`).
`HybridRetriever` degrades gracefully to pure BM25 if the dense model
isn't available, rather than crashing a sweep partway through.

## Reranking and query transforms

Two more ablation axes sit alongside chunking/retrieval/top-k:

**Reranking** (`rag/rerank.py`) is a second pass over the retriever's
top candidates, scored jointly with the query instead of independently:

```bash
python experiments/run_ablation.py --generator ollama --ollama-model llama3.2 \
    --rerankers none lexical_overlap cross_encoder
```

`cross_encoder` downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` on
first use; `lexical_overlap` is a dependency-free baseline so
"does reranking help at all" stays testable offline.

**Query transforms** (`rag/query_transform.py`) rewrite the query
*before* first-stage retrieval:

- `hyde` -- ask the generator for a hypothetical answer passage, then
  retrieve using that instead of the (often lexically sparse) question.
- `multi_query` -- generate several phrasings of the question, retrieve
  for each, fuse via reciprocal rank fusion (RAG-Fusion style).
- `decompose` -- split a compound question into atomic sub-questions,
  retrieve per sub-question, merge -- targeting multi-hop questions
  whose answer spans more than one document.

```bash
python experiments/run_ablation.py --generator ollama --ollama-model llama3.2 \
    --query-transforms identity hyde multi_query decompose
```

All three need a real generator to do the actual rewriting -- with
`--generator mock` they fall back to deterministic, non-model behavior
(documented in each class's docstring) so a `--quick`-style offline run
still exercises the code path without pretending a mock model can write
a hypothetical passage.

## LLM-judge scoring in the ablation sweep

`--use-llm-judge` scores each answer's faithfulness and relevance with
the same generator (via `eval/judge.py`'s `LLMJudge`), in addition to
the always-on lexical faithfulness fallback:

```bash
python experiments/run_ablation.py --generator ollama --ollama-model llama3.2 \
    --use-llm-judge
```

This adds two extra generation calls per query per config, so it's
meaningfully slower than the lexical-only default -- opt in once you're
past pipeline sanity-checking.

## Is that difference real? (`eval/significance.py`)

The ablation CSV's per-config means invite reading noise as signal on a
small eval set. `compare_configs()` gives a paired, nonparametric answer:

```python
import csv
from eval.significance import compare_configs

with open("experiments/results/ablation_results.csv") as f:
    rows = list(csv.DictReader(f))

rows_a = [r for r in rows if r["retriever"] == "bm25" and r["reranker"] == "none"]
rows_b = [r for r in rows if r["retriever"] == "bm25" and r["reranker"] == "cross_encoder"]
for r in rows_a + rows_b:
    r["recall@3"] = float(r["recall@3"])

result = compare_configs(rows_a, rows_b, metric="recall@3")
print(result.summary())
```

Uses a bootstrap confidence interval per config and a paired permutation
test on the shared query_ids -- appropriate for the 8-50 labeled queries
typical of a hand-built eval set, where a normal-approximation t-test's
assumptions are shaky.

## How much to trust the judge (`eval/judge_calibration.py`)

The known confound stated below (same small model as generator and
judge) is measurable, not just a caveat. Hand-label a sample of judge
outputs (see `data/sample_corpus/sample_judge_calibration.json` for the
format -- replace the placeholder values with real ratings of your own
judge's outputs before drawing any conclusion from it) and run:

```python
from eval.judge_calibration import load_human_labels, calibration_report

human_labels = load_human_labels("data/sample_corpus/sample_judge_calibration.json")
judge_scores = {"q1": 5, "q2": 4, "q3": 4, "q4": 2, "q8": 5}  # from your ablation run
report = calibration_report(judge_scores, human_labels, metric="faithfulness")
print(report.summary())
```

Reports linear-weighted Cohen's kappa (chance-corrected agreement,
appropriate for an ordinal 1-5 scale), mean absolute error, and
exact/within-one-point agreement rates.

## Design notes

**Why separate retrieval and generation metrics?** A RAG system's failure
mode is only actionable if you know which stage failed. High faithfulness
with low retrieval recall usually means the generator is *correctly*
refusing to answer from insufficient context — that's not a generation
bug. Low faithfulness with high retrieval precision means the generator
is ignoring good context and hallucinating anyway — that's a prompting or
model-choice problem, not a retrieval one. Reporting one blended "RAG
score" erases this distinction.

**Why an adversarial no-answer query?** Most demo eval sets only contain
answerable questions, which can't distinguish "the model answers well"
from "the model always answers, and happens to be right when there's
relevant context." A query with genuinely no supporting document in the
corpus tests refusal behavior directly.

**Why a lexical-overlap faithfulness fallback alongside LLM-as-judge?**
LLM-as-judge is the standard approach (see RAGAS, ARES) but a small local
judge model is noisier than GPT-4-class judges used in most published
faithfulness benchmarks. The lexical overlap metric is a weaker but
free, deterministic, always-available second signal — useful as a sanity
check when judge scores look surprising, not a replacement for it.

**What's a known limitation of this design?** Using the same small local
model as both generator and judge is a real confound (a model may be
systematically lenient toward its own outputs). If you have access to a
stronger model for judging even occasionally, using it to spot-check a
sample of judge scores is worth doing and reporting on in
`reports/findings.md`.

## Running tests

```bash
python -m pytest tests/ -v
```

All tests run fully offline (chunking logic, BM25 retrieval ranking,
reranking, query transform fallbacks, eval metrics, bootstrap/permutation
significance testing, and judge-calibration agreement stats) — no model
downloads required.

## Roadmap / extensions

Done:
- [x] Reranker stage (cross-encoder, + offline lexical-overlap fallback) as an ablation axis
- [x] Multi-hop query decomposition
- [x] Query rewriting / HyDE and multi-query (RAG-Fusion) as retrieval-quality interventions
- [x] Cost/latency tracking alongside quality metrics (`estimated_tokens`, `estimated_cost_usd`, split retrieval/generation latency)
- [x] Statistical significance testing between configs (bootstrap CI + paired permutation test)
- [x] Judge-calibration tooling (weighted kappa, MAE vs. human labels) — still needs a real human-labeled set to be meaningful; the bundled file is a format example, not real data

Still open:
- [ ] Cost/latency tradeoff *curves* (plotting, not just the raw per-row numbers -- pandas/matplotlib are already optional deps for this)
- [ ] A real human-annotated judge-calibration set of non-trivial size (the bundled 8-example file is a format demo)
- [ ] Wire a paid API generator in behind `BaseGenerator` so `cost_per_1k_tokens` has a non-hypothetical use case
