from typing import List, Dict


def reciprocal_rank_fusion(result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    """Combine ranked result lists using Reciprocal Rank Fusion (RRF).

    Each entry in `result_lists` is a list of documents represented as dicts
    with at least an `id` key. Returns a fused list of documents ordered by
    fused score (descending).
    """
    scores: Dict[str, float] = {}
    documents: Dict[str, Dict] = {}

    for results in result_lists:
        for rank, document in enumerate(results, start=1):
            doc_id = document.get("id")
            if doc_id is None:
                continue

            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            documents[doc_id] = document

    ranked_ids = sorted(scores.keys(), key=lambda _id: scores[_id], reverse=True)
    return [documents[doc_id] for doc_id in ranked_ids]
