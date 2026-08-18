from typing import List, Dict
from collections import defaultdict


def reciprocal_rank_fusion(result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    """Reciprocal Rank Fusion (RRF).

    result_lists: list of ranked lists (best->worst). Each item must have an
    `id` key and may include `score`, `text`, and `metadata`.

    Returns fused ranked list where each result dict contains the original
    fields plus an added `rrf_score` field.
    """
    scores = defaultdict(float)
    documents: Dict[str, Dict] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            doc_id = result.get("id")
            if not doc_id:
                continue

            scores[doc_id] += 1.0 / (k + rank)

            # store representative document (do not mutate scores here)
            if doc_id not in documents:
                documents[doc_id] = result.copy()

    ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    fused_results: List[Dict] = []
    for doc_id in ranked_ids:
        result = documents[doc_id].copy()
        result["rrf_score"] = scores[doc_id]
        fused_results.append(result)

    return fused_results


# Backwards-compatible alias
def rrf_fuse(result_lists: List[List[Dict]], k: int = 60, top_n: int = 10) -> List[Dict]:
    fused = reciprocal_rank_fusion(result_lists, k=k)
    return fused[:top_n]
