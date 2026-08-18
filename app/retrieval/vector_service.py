"""Normalize vector retrieval results to a consistent chunk-based format.

Provides a small wrapper around `vector_store.similarity_search` that returns
results as a list of dicts with keys: `id`, `text`, `score`, `metadata` where
`metadata` contains `chunk_id`.
"""
from typing import List, Dict, Any

from app.retrieval.vector_store import vector_store


def search(query: str, k: int = 10) -> List[Dict[str, Any]]:
    documents = vector_store.similarity_search(query, k=k)
    results: List[Dict[str, Any]] = []

    for doc in documents:
        meta = getattr(doc, "metadata", {}) or {}
        chunk_id = meta.get("chunk_id") or meta.get("orig_id")

        # Ensure the metadata contains `chunk_id` for downstream consistency
        if chunk_id and "chunk_id" not in meta:
            try:
                meta["chunk_id"] = chunk_id
            except Exception:
                pass

        results.append(
            {
                "id": chunk_id or "",
                "text": getattr(doc, "page_content", getattr(doc, "text", "")),
                "score": meta.get("score"),
                "metadata": meta,
            }
        )

    return results
