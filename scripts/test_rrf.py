from app.retrieval.bm25_service import bm25_retriever
from app.retrieval.vector_store import vector_store
from app.retrieval.rrf import reciprocal_rank_fusion


def main():
    query = "How does Kafka consumer rebalancing work?"

    vector_documents = vector_store.similarity_search(
        query,
        k=10,
    )

    vector_results = [
        {
            "id": doc.metadata.get("chunk_id") or doc.metadata.get("orig_id"),
            "text": getattr(doc, "page_content", getattr(doc, "text", "")),
            "score": doc.metadata.get("score"),
            "metadata": dict(doc.metadata),
        }
        for doc in vector_documents
    ]

    bm25_results = bm25_retriever.search(
        query,
        k=10,
    )

    fused_results = reciprocal_rank_fusion(
        [
            vector_results,
            bm25_results,
        ],
        k=60,
    )

    print("\n=== VECTOR ===")
    for rank, result in enumerate(vector_results, start=1):
        print(rank, result["id"])

    print("\n=== BM25 ===")
    for rank, result in enumerate(bm25_results, start=1):
        print(rank, result["id"])

    print("\n=== RRF ===")
    for rank, result in enumerate(fused_results[:10], start=1):
        print(rank, result["id"], result.get("rrf_score"))


if __name__ == "__main__":
    main()
