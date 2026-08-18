import json
import time
from pathlib import Path
from typing import List, Dict, Any

from evaluation.metrics.retrieval import recall_at_k, reciprocal_rank, mean
from app.retrieval.vector_store import vector_store
from app.retrieval.bm25_service import bm25_retriever
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval import config as cfg

DS_PATH = Path("evaluation/datasets/golden_dataset.jsonl")
OUT_DIR = Path("evaluation/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Asymmetric configurations requested
CONFIGS = [
    (10, 20, 60, 5),
    (10, 50, 60, 5),
    (20, 50, 60, 5),
    (20, 50, 30, 5),
    (10, 50, 30, 5),
    (20, 20, 60, 5),
]


def normalize_vector_docs(docs: List[Any]) -> List[Dict[str, Any]]:
    out = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        out.append(
            {
                "id": meta.get("chunk_id") or meta.get("orig_id") or meta.get("id") or getattr(doc, "id", ""),
                "text": getattr(doc, "page_content", getattr(doc, "text", "")),
                "vector_score": meta.get("score"),
                "metadata": dict(meta),
            }
        )
    return out


def run_config(vec_k: int | None, bm_k: int | None, rrf_k: int | None, final_k: int | None) -> Dict[str, Any]:
    per_query = []
    recall5 = []
    rr_vals = []
    latencies = []
    per_cat = {}

    with DS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            tc = json.loads(line)
            qid = tc["id"]
            qcat = tc.get("category", "unknown")
            query = tc["query"]
            relevant = tc.get("relevant_chunk_ids", [])

            # use config defaults when None
            _vec_k = vec_k if vec_k is not None else cfg.VECTOR_K
            _bm_k = bm_k if bm_k is not None else cfg.BM25_K
            _rrf_k = rrf_k if rrf_k is not None else cfg.RRF_K
            _final_k = final_k if final_k is not None else cfg.FINAL_K

            v_start = time.perf_counter()
            v_docs = vector_store.similarity_search(query, k=_vec_k)
            v_lat = (time.perf_counter() - v_start) * 1000.0
            v_results = normalize_vector_docs(v_docs)

            b_start = time.perf_counter()
            b_results = bm25_retriever.search(query, k=_bm_k)
            b_lat = (time.perf_counter() - b_start) * 1000.0

            fuse_start = time.perf_counter()
            fused_all = reciprocal_rank_fusion([v_results, b_results], k=_rrf_k)
            fuse_lat = (time.perf_counter() - fuse_start) * 1000.0

            fused = fused_all[:_final_k]
            retrieved_ids = [r.get("id") for r in fused]

            r5 = recall_at_k(retrieved_ids, relevant, 5)
            rr = reciprocal_rank(retrieved_ids, relevant)

            per_query.append({"id": qid, "category": qcat, "recall@5": r5, "reciprocal_rank": rr})
            recall5.append(r5)
            rr_vals.append(rr)
            latencies.append(v_lat + b_lat + fuse_lat)
            per_cat.setdefault(qcat, []).append(r5)

    result = {
        "vec_k": vec_k,
        "bm_k": bm_k,
        "rrf_k": rrf_k,
        "final_k": final_k,
        "num_queries": len(per_query),
        "overall_recall@5": float(mean(recall5)),
        "mrr": float(mean(rr_vals)),
        "avg_latency_ms": float(mean(latencies)),
        "per_category_recall@5": {k: float(sum(v) / len(v)) for k, v in per_cat.items()},
    }
    return result


def main():
    all_results = []
    for vec_k, bm_k, rrf_k, final_k in CONFIGS:
        print(f"Running vec_k={vec_k} bm_k={bm_k} rrf_k={rrf_k} final_k={final_k}")
        res = run_config(vec_k, bm_k, rrf_k, final_k)
        all_results.append(res)

    out = OUT_DIR / "rrf_asymmetric_grid.json"
    out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Wrote results to {out}\n")

    # print summary table
    cats = sorted({c for r in all_results for c in r["per_category_recall@5"].keys()})
    header = "| vec_k | bm_k | rrf_k | final_k | overall@5 | mrr | avg_latency_ms | " + " | ".join(cats) + " |"
    print(header)
    print("|---:|---:|---:|---:|---:|---:|---:|" + "---:|" * len(cats))
    for r in all_results:
        row = f"| {r['vec_k']} | {r['bm_k']} | {r['rrf_k']} | {r['final_k']} | {r['overall_recall@5']:.3f} | {r['mrr']:.3f} | {r['avg_latency_ms']:.1f} | "
        row += " | ".join(f"{r['per_category_recall@5'].get(c,0.0):.3f}" for c in cats)
        row += " |"
        print(row)


if __name__ == '__main__':
    main()
