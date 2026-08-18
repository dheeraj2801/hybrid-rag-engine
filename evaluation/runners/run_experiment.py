import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from evaluation.metrics.retrieval import recall_at_k, reciprocal_rank, mean

from app.retrieval.vector_store import vector_store
from app.retrieval.bm25_service import bm25_retriever
from app.retrieval.rrf import reciprocal_rank_fusion


RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DS_PATH = Path("evaluation/datasets/golden_dataset.jsonl")


def extract_doc_id(doc: Any) -> str:
    # support dict-like BM25 results and langchain Documents
    if isinstance(doc, dict):
        return doc.get("id") or doc.get("chunk_id") or doc.get("orig_id") or ""

    meta = getattr(doc, "metadata", {}) or {}
    if "orig_id" in meta:
        return meta["orig_id"]
    if "chunk_id" in meta:
        return meta["chunk_id"]
    if "id" in meta:
        return meta["id"]
    if "source" in meta:
        return meta["source"]
    return getattr(doc, "id", "") or getattr(doc, "_id", "") or ""


def normalize_vector_docs(docs: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        out.append(
            {
                "id": meta.get("chunk_id") or meta.get("orig_id") or meta.get("id") or extract_doc_id(doc),
                "text": getattr(doc, "page_content", getattr(doc, "text", "")),
                "vector_score": meta.get("score"),
                "metadata": dict(meta),
            }
        )
    return out


def run_vector_baseline(k: int = 10):
    results = []

    recall_scores = {1: [], 5: [], 10: []}
    rr_scores = []
    latencies = []

    with DS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            tc = json.loads(line)
            query = tc["query"]
            relevant = tc.get("relevant_chunk_ids", [])

            start = time.perf_counter()
            docs = vector_store.similarity_search(query, k=k)
            latency_ms = (time.perf_counter() - start) * 1000.0

            retrieved_ids = [extract_doc_id(d) for d in docs]

            r1 = recall_at_k(retrieved_ids, relevant, 1)
            r5 = recall_at_k(retrieved_ids, relevant, 5)
            r10 = recall_at_k(retrieved_ids, relevant, 10)
            rr = reciprocal_rank(retrieved_ids, relevant)

            recall_scores[1].append(r1)
            recall_scores[5].append(r5)
            recall_scores[10].append(r10)
            rr_scores.append(rr)
            latencies.append(latency_ms)

            results.append(
                {
                    "id": tc["id"],
                    "query": query,
                    "retrieved_ids": retrieved_ids,
                    "relevant_ids": relevant,
                    "recall@1": r1,
                    "recall@5": r5,
                    "recall@10": r10,
                    "reciprocal_rank": rr,
                    "latency_ms": latency_ms,
                }
            )

    output = {
        "mode": "vector",
        "num_queries": len(results),
        "aggregate": {
            "recall@1": mean(recall_scores[1]),
            "recall@5": mean(recall_scores[5]),
            "recall@10": mean(recall_scores[10]),
            "mrr": mean(rr_scores),
            "avg_latency_ms": mean(latencies),
        },
        "per_query": results,
    }

    out_path = RESULTS_DIR / "vector.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote results to {out_path}")


def run_bm25_baseline(k: int = 10):
    results = []

    recall_scores = {1: [], 5: [], 10: []}
    rr_scores = []
    latencies = []

    with DS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            tc = json.loads(line)
            query = tc["query"]
            relevant = tc.get("relevant_chunk_ids", [])

            start = time.perf_counter()
            docs = bm25_retriever.search(query, k=k)
            latency_ms = (time.perf_counter() - start) * 1000.0

            retrieved_ids = [extract_doc_id(d) for d in docs]

            r1 = recall_at_k(retrieved_ids, relevant, 1)
            r5 = recall_at_k(retrieved_ids, relevant, 5)
            r10 = recall_at_k(retrieved_ids, relevant, 10)
            rr = reciprocal_rank(retrieved_ids, relevant)

            recall_scores[1].append(r1)
            recall_scores[5].append(r5)
            recall_scores[10].append(r10)
            rr_scores.append(rr)
            latencies.append(latency_ms)

            results.append(
                {
                    "id": tc["id"],
                    "query": query,
                    "retrieved_ids": retrieved_ids,
                    "relevant_ids": relevant,
                    "recall@1": r1,
                    "recall@5": r5,
                    "recall@10": r10,
                    "reciprocal_rank": rr,
                    "latency_ms": latency_ms,
                }
            )

    output = {
        "mode": "bm25",
        "num_queries": len(results),
        "aggregate": {
            "recall@1": mean(recall_scores[1]),
            "recall@5": mean(recall_scores[5]),
            "recall@10": mean(recall_scores[10]),
            "mrr": mean(rr_scores),
            "avg_latency_ms": mean(latencies),
        },
        "per_query": results,
    }

    out_path = RESULTS_DIR / "bm25.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote results to {out_path}")


def run_rrf(k: int = 10, rrf_k: int = 60):
    results = []

    recall_scores = {1: [], 5: [], 10: []}
    rr_scores = []
    latencies = []

    with DS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            tc = json.loads(line)
            query = tc["query"]
            relevant = tc.get("relevant_chunk_ids", [])

            # run vector
            vec_start = time.perf_counter()
            vec_docs = vector_store.similarity_search(query, k=k)
            vec_latency = (time.perf_counter() - vec_start) * 1000.0
            vec_results = normalize_vector_docs(vec_docs)

            # run bm25
            bm_start = time.perf_counter()
            bm_results = bm25_retriever.search(query, k=k)
            bm_latency = (time.perf_counter() - bm_start) * 1000.0

            start = time.perf_counter()
            fused = reciprocal_rank_fusion([vec_results, bm_results], k=rrf_k)
            latency_ms = (time.perf_counter() - start) * 1000.0

            # build retrieved ids and compute metrics
            retrieved_ids = [r.get("id") for r in fused]

            r1 = recall_at_k(retrieved_ids, relevant, 1)
            r5 = recall_at_k(retrieved_ids, relevant, 5)
            r10 = recall_at_k(retrieved_ids, relevant, 10)
            rr = reciprocal_rank(retrieved_ids, relevant)

            recall_scores[1].append(r1)
            recall_scores[5].append(r5)
            recall_scores[10].append(r10)
            rr_scores.append(rr)
            latencies.append(vec_latency + bm_latency + latency_ms)

            results.append(
                {
                    "id": tc["id"],
                    "query": query,
                    "retrieved": fused,
                    "retrieved_ids": retrieved_ids,
                    "relevant_ids": relevant,
                    "recall@1": r1,
                    "recall@5": r5,
                    "recall@10": r10,
                    "reciprocal_rank": rr,
                    "latency_ms": vec_latency + bm_latency + latency_ms,
                }
            )

    output = {
        "mode": "rrf",
        "num_queries": len(results),
        "params": {"rrf_k": rrf_k},
        "aggregate": {
            "recall@1": mean(recall_scores[1]),
            "recall@5": mean(recall_scores[5]),
            "recall@10": mean(recall_scores[10]),
            "mrr": mean(rr_scores),
            "avg_latency_ms": mean(latencies),
        },
        "per_query": results,
    }

    out_path = RESULTS_DIR / "rrf.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote results to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vector", "bm25", "rrf", "all"], default="all")
    parser.add_argument("--k", type=int, default=10, help="top-k per retriever")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF k parameter")
    args = parser.parse_args()

    if args.mode in ("vector", "all"):
        run_vector_baseline(k=args.k)
    if args.mode in ("bm25", "all"):
        run_bm25_baseline(k=args.k)
    if args.mode in ("rrf", "all"):
        run_rrf(k=args.k, rrf_k=args.rrf_k)


if __name__ == "__main__":
    main()
