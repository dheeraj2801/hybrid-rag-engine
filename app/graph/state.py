from typing import TypedDict


class RAGState(TypedDict):
    query: str
    documents: list
    context: str
    answer: str