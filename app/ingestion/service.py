"""Shared ingestion service utilities.

Provides a `load_chunks` convenience function that the vector and BM25
ingestion paths can both reuse to ensure chunk IDs remain consistent.
"""
from typing import List, Dict

from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents


def load_chunks(data_path: str = "data") -> List[Dict]:
    """Load documents from `data_path`, chunk them, and return chunk dicts.

    Each chunk is a dict with keys: `id`, `text`, `source`, `parent_id`.
    """
    documents = load_documents(data_path)
    return chunk_documents(documents)
