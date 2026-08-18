"""BM25 service singleton.

Builds a BM25Retriever from the current corpus once at import time so the
index can be reused for every query during the application's lifetime.
"""
from app.ingestion.service import load_chunks
from app.retrieval.bm25_store import BM25Retriever


# Load and build BM25 index at import time (singleton)
chunks = load_chunks()
bm25_retriever = BM25Retriever(chunks)


def get_bm25_retriever() -> BM25Retriever:
    """Return the singleton BM25 retriever."""
    return bm25_retriever
