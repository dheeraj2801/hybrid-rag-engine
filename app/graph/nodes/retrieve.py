from app.retrieval.vector_service import search as vector_search


def retrieve(state):
    results = vector_search(state["query"], k=5)

    return {
        "documents": results,
    }