from app.retrieval.vector_store import vector_store


def main():
    query = "How does Kafka consumer rebalancing work?"

    results = vector_store.similarity_search(
        query,
        k=5,
    )

    for i, document in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print(document.page_content)
        print(document.metadata)


if __name__ == "__main__":
    main()