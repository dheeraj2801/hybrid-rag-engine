import json
import time
from pathlib import Path
from typing import List, Dict, Any

from evaluation.metrics.retrieval import recall_at_k, mean
from app.retrieval.vector_store import vector_store
from app.retrieval.bm25_service import bm25_retriever
from app.retrieval.rrf import reciprocal_rank_fusion

DS_PATH = Path("evaluation/datasets/golden_dataset.jsonl")
OUT_DIR = Path("evaluation/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# tuning configurations: (vec_k, bm_k, rrf_k, final_k)
CONFIGS = [
    (10, 10, 60, 5),
    (20, 20, 60, 5),
    (50, 50, 60, 5),
    (10, 10, 10, 5),
    (10, 10, 30, 5),
    (100, 100, 60, 5),
]


def extract_doc_id(result: Any) -> str:
    if isinstance(result, dict):
        return result.get("id") or result.get("chunk_id") or result.get("orig_id") or ""
    meta = getattr(result, "metadata", {}) or {}
    return meta.get("chunk_id") or meta.get("orig_id") or meta.get("id") or getattr(result, "id", "")


def normalize_vector_docs(docs: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        out.append({
            "id": meta.get("chunk_id") or meta.get("orig_id") or meta.get("id") or extract_doc_id(doc),
            "text": getattr(doc, "page_content", getattr(doc, "text", "")),
            "vector_score": meta.get("score"),
            "metadata": dict(meta),
        })
    return out


all_results = []

# load dataset into memory
dataset = [json.loads(line) for line in DS_PATH.open("r", encoding="utf-8")]

for vec_k, bm_k, rrf_k, final_k in CONFIGS:
    print(f"Running config vec_k={vec_k} bm_k={bm_k} rrf_k={rrf_k} final_k={final_k}")
    per_query = []
    per_cat = {}
    for tc in dataset:
        qid = tc["id"]
        qcat = tc.get("category", "unknown")
        query = tc["query"]
        relevant = tc.get("relevant_chunk_ids", [])

        # vector
        v_start = time.perf_counter()
        v_docs = vector_store.similarity_search(query, k=vec_k)
        v_lat = (time.perf_counter() - v_start) * 1000.0
        v_results = normalize_vector_docs(v_docs)

        # bm25
        b_start = time.perf_counter()
        b_results = bm25_retriever.search(query, k=bm_k)
        b_lat = (time.perf_counter() - b_start) * 1000.0

        fused_all = reciprocal_rank_fusion([v_results, b_results], k=rrf_k)
        fused = fused_all[:final_k]

        retrieved_ids = [r.get("id") for r in fused]
        r5 = recall_at_k(retrieved_ids, relevant, 5)

        per_query.append({
            "id": qid,
            "category": qcat,
            "recall@5": r5,
            "retrieved_ids": retrieved_ids,
        })

        per_cat.setdefault(qcat, []).append(r5)

    # aggregate
    cat_means = {cat: float(sum(vals) / len(vals)) for cat, vals in per_cat.items()}
    overall = float(sum([p["recall@5"] for p in per_query]) / len(per_query))

    result = {
        "vec_k": vec_k,
        "bm_k": bm_k,
        "rrf_k": rrf_k,
        "final_k": final_k,
        "overall_recall@5": overall,
        "per_category_recall@5": cat_means,
    }

    all_results.append(result)

# write results
out_path = OUT_DIR / "rrf_tuning.json"
out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
print(f"Wrote tuning results to {out_path}")

# print summary table focused on error_code and overall
print("\n| vec_k | bm_k | rrf_k | final_k | overall@5 | error_code@5 |")
print("|---:|---:|---:|---:|---:|---:|")
for r in all_results:
    err = r["per_category_recall@5"].get("error_code", 0.0)
    print(f"| {r['vec_k']} | {r['bm_k']} | {r['rrf_k']} | {r['final_k']} | {r['overall_recall@5']:.3f} | {err:.3f} |")
