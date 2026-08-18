import json
import time
from pathlib import Path
from typing import List, Dict, Any

from evaluation.metrics.retrieval import recall_at_k
from app.retrieval.vector_store import vector_store
from app.retrieval.bm25_service import bm25_retriever
from app.retrieval.rrf import reciprocal_rank_fusion

DS_PATH = Path("evaluation/datasets/golden_dataset.jsonl")

from app.retrieval import config as cfg

vec_k = None
bm_k = None
rrf_k = None
final_k = None

dataset = [json.loads(line) for line in DS_PATH.open("r", encoding="utf-8")]

cases = []
for tc in dataset:
    if tc.get("category") != "error_code":
        continue
    qid = tc["id"]
    query = tc["query"]
    relevant = tc.get("relevant_chunk_ids", [])

    _vec_k = vec_k if vec_k is not None else cfg.VECTOR_K
    _bm_k = bm_k if bm_k is not None else cfg.BM25_K
    _rrf_k = rrf_k if rrf_k is not None else cfg.RRF_K
    _final_k = final_k if final_k is not None else cfg.FINAL_K

    v_docs = vector_store.similarity_search(query, k=_vec_k)
    v_results = []
    for d in v_docs:
        meta = getattr(d, "metadata", {}) or {}
        v_results.append({"id": meta.get("chunk_id") or meta.get("orig_id") or meta.get("id")})

    b_results = bm25_retriever.search(query, k=_bm_k)

    fused_all = reciprocal_rank_fusion([v_results, b_results], k=_rrf_k)
    fused = fused_all[:_final_k]

    bm_ids = [r.get("id") for r in b_results]
    fused_ids = [r.get("id") for r in fused]

    bm_r5 = recall_at_k(bm_ids, relevant, 5)
    rrf_r5 = recall_at_k(fused_ids, relevant, 5)

    if bm_r5 > 0 and rrf_r5 == 0:
        cases.append({
            "id": qid,
            "query": query,
            "relevant": relevant,
            "bm25_top5": bm_ids[:5],
            "rrf_top5": fused_ids,
        })

out = Path("evaluation/results/rrf_error_code_failures.json")
out.write_text(json.dumps(cases, indent=2), encoding="utf-8")
print(f"Wrote {len(cases)} failing cases to {out}")
for c in cases:
    print("---")
    print(c["id"], c["query"])
    print("relevant:", c["relevant"]) 
    print("bm25_top5:", c["bm25_top5"]) 
    print("rrf_top5:", c["rrf_top5"]) 
