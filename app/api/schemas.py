from pydantic import BaseModel
from typing import List


class PingResponse(BaseModel):
    status: str


class QueryRequest(BaseModel):
    query: str
    retrieval_mode: str = "hybrid_rerank"
    top_k: int = 5


class Source(BaseModel):
    document_id: str
    score: float
    content: str


class RetrievalMeta(BaseModel):
    mode: str
    vector_candidates: int
    bm25_candidates: int
    rrf_candidates: int
    reranked_candidates: int


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    retrieval: RetrievalMeta
