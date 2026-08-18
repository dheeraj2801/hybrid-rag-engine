from app.retrieval.bm25_service import bm25_retriever


def main():
    query = "How does Kafka consumer rebalancing work?"
    results = bm25_retriever.search(
        query,
        k=5,
    )

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"ID: {result['id']}")
        print(f"Score: {result['score']}")
        print(f"Source: {result['metadata']['source']}")
        print(result["text"]) 


if __name__ == "__main__":
    main()
