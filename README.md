
# Hybrid RAG Benchmark

Production-oriented RAG system comparing semantic search, BM25, RRF fusion, and cross-encoder reranking using FastAPI and LangGraph.

Architecture

```
			    ┌───────────────┐
			    │    FastAPI    │
			    │   /query      │
			    └───────┬───────┘
				     │
				     ▼
			    ┌───────────────┐
			    │   LangGraph   │
			    └───────┬───────┘
				     │
			  ┌─────────┴─────────┐
			  │                   │
			  ▼                   ▼
		  Semantic Search         BM25
			  │                   │
			  └─────────┬─────────┘
				     │
				     ▼
				  RRF
				     │
				     ▼
			     Candidate Set
				     │
				     ▼
			    Cross Encoder
			      Reranking
				     │
				     ▼
			     Top-K Context
				     │
				     ▼
				   LLM
				     │
				     ▼
				  Answer
```

Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# run app
uvicorn app.main:app --reload --port 8000
```

API

POST `/query` accepts a JSON payload with `query`, `retrieval_mode`, and `top_k` and returns an `answer`, `sources`, and `retrieval` metadata.

Model caching

The embedding and transformer weights can be large and are downloaded the
first time a model is instantiated. To avoid downloading on service startup
and to cache the weights ahead of time, run:

```bash
# set HF token for authenticated downloads (recommended)
export HF_TOKEN=your_hf_token_here
python scripts/preload_models.py
```

This will download the model weights into your HF cache so the app starts
without heavy downloads.

## Evaluation — Retrieval Baselines

The repository includes a small evaluation harness to measure retrieval quality
for multiple retrieval modes using the same golden dataset.

Files:

- `evaluation/datasets/golden_dataset.jsonl` — the golden dataset (20 queries).
 - `evaluation/datasets/golden_dataset.jsonl` — the golden dataset (150 queries).
- `evaluation/runners/run_experiment.py` — runner for experiments (supports `--mode vector|bm25|rrf|all`).
- `evaluation/metrics/retrieval.py` — retrieval metric implementations.
- `evaluation/results/vector.json`, `evaluation/results/bm25.json`, `evaluation/results/rrf.json` — saved results from the last run.

Run the experiments:

```bash
source .venv/bin/activate
python evaluation/runners/run_experiment.py --mode all
```

Aggregate results (last run on the 150-query golden dataset)

| Metric | Vector (semantic) | BM25 (lexical) | RRF (fusion) |
|---|---:|---:|---:|
| Queries | 150 | 150 | 150 |
| Recall@1 | 0.04 | 0.053 | 0.067 |
| Recall@5 | 0.28 | 0.517 | 0.483 |
| Recall@10 | 0.483 | 0.653 | 0.890 |
| MRR | 0.141 | 0.221 | 0.272 |
| Avg latency (ms) | 21.69 | 0.055 | 20.73 |

Per-query details and the full retrieved candidate lists are saved in the `evaluation/results` JSON files linked above. Use the same golden dataset to run additional modes (hybrid, reranker) for fair comparisons.

Next steps:

- Add cross-encoder reranking and measure accuracy vs latency tradeoffs.
- Wire RRF into the live LangGraph pipeline for hybrid retrieval (parallel retrieval → RRF → reranker → LLM).
- Add a generation-scoring step (LLM-as-judge) to evaluate final answer correctness.

**Retrieval Baseline (frozen)**

Evaluated on 150 queries across five categories: semantic, exact, technical, error_code, multi_concept.

Best tuned RRF configuration (recorded as the baseline):

- Vector candidates: 20
- BM25 candidates: 50
- RRF k: 60
- Final candidates: 5

Aggregate metrics (Recall@5): **0.540**

Per-category Recall@5:

| Query Type | Recall@5 |
|---|---:|
| semantic | 0.567 |
| exact | 0.633 |
| technical | 0.633 |
| error_code | 0.467 |
| multi_concept | 0.400 |

Notes: This baseline was chosen to maximize general-purpose retrieval performance across categories rather than optimizing for a single category. See `evaluation/results/rrf_asymmetric_grid.json` for the full sweep.

Runner defaults

All evaluation runners (`evaluation/runners/*.py`) now use the frozen retrieval baseline defined in `app/retrieval/config.py` by default (`VECTOR_K`, `BM25_K`, `RRF_K`, `FINAL_K`). Each runner still exposes CLI flags to override these values for experiments, for example:

```bash
# use frozen defaults
python evaluation/runners/run_experiment.py --mode all

# override BM25 depth only
python evaluation/runners/run_experiment.py --mode rrf --bm25-k 100
```

Per-category Recall@5 (expanded 150-query dataset)

| Query Type | Vector Recall@5 | BM25 Recall@5 | RRF Recall@5 |
|---|---:|---:|---:|
| semantic | 0.467 | 0.500 | 0.533 |
| exact | 0.200 | 0.667 | 0.667 |
| technical | 0.133 | 0.500 | 0.500 |
| error_code | 0.333 | 0.667 | 0.333 |
| multi_concept | 0.267 | 0.250 | 0.383 |

Notes: dataset expanded to 150 queries (30 per category). Results suggest BM25 dominates on exact-keyword queries, vector search performs well on semantic queries, and RRF improves mixed/multi-concept queries by combining strengths of both methods. Treat these numbers as illustrative; run more extensive sweeps (different `k`, reranker inclusion) before drawing production conclusions.


