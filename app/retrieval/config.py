"""Centralized retrieval baseline configuration.

This file records the frozen hybrid retrieval baseline chosen after
asymmetric grid tuning on the 150-query golden dataset.
"""

# Number of vector candidates to retrieve per query
VECTOR_K = 20

# Number of BM25 candidates to retrieve per query
BM25_K = 50

# Reciprocal Rank Fusion smoothing constant
RRF_K = 60

# Final number of candidates to return to downstream reranker/LLM
FINAL_K = 5


DEFAULTS = {
    "vector_k": VECTOR_K,
    "bm25_k": BM25_K,
    "rrf_k": RRF_K,
    "final_k": FINAL_K,
}
