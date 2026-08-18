
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

## Evaluation — Vector Search Baseline

The repository includes a small evaluation harness to measure retrieval quality
for the vector-only baseline. Files:

- `evaluation/datasets/golden_dataset.jsonl` — the golden dataset (20 queries).
- `evaluation/runners/run_experiment.py` — runner for baseline experiments.
- `evaluation/metrics/retrieval.py` — retrieval metric implementations.
- `evaluation/results/vector.json` — saved results from the last run.

Run the vector baseline:

```bash
source .venv/bin/activate
python evaluation/runners/run_experiment.py
```

Results (vector baseline):

| Metric | Value |
|---|---:|
| Queries | 20 |
| Recall@1 | 0.45 |
| Recall@5 | 0.925 |
| Recall@10 | 1.00 |
| MRR | 0.6646 |
| Avg latency (ms) | 61.80 |

The per-query results and detailed retrieval lists are saved to
`evaluation/results/vector.json`. Use the same golden dataset to run
other retrieval modes (hybrid, RRF, reranker) for fair comparisons.

Next steps:

- Add BM25 / hybrid retrieval and run the same experiment.
- Implement cross-encoder reranking and measure accuracy vs latency tradeoffs.
- Add a generation-scoring step (LLM-as-judge) to evaluate final answer correctness.


