from typing import List


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    retrieved_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    return len(retrieved_k & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    relevant = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def mean(scores: List[float]) -> float:
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
