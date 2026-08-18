
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

