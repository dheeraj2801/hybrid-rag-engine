from app.retrieval.vector_store import vector_store


def retrieve(state):

    results = vector_store.similarity_search(
        state["query"],
        k=5,
    )

    return {
        "documents": results,
    }