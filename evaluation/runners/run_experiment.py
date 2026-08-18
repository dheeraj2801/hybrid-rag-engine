import json
import time
from pathlib import Path
from typing import List

from evaluation.metrics.retrieval import recall_at_k, reciprocal_rank, mean

from app.retrieval.vector_store import vector_store


RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_doc_id(doc) -> str:
    # langchain Documents should have .metadata with our stored fields.
    meta = getattr(doc, "metadata", {}) or {}
    if "orig_id" in meta:
        return meta["orig_id"]
    if "id" in meta:
        return meta["id"]
    if "source" in meta:
        return meta["source"]
    return getattr(doc, "id", "") or getattr(doc, "_id", "") or ""


def run_vector_baseline(k: int = 10):
    ds_path = Path("evaluation/datasets/golden_dataset.jsonl")
    results = []

    recall_scores = {1: [], 5: [], 10: []}
    rr_scores = []
    latencies = []

    with ds_path.open("r", encoding="utf-8") as fh:
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


if __name__ == "__main__":
    run_vector_baseline()
