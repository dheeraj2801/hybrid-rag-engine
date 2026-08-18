from fastapi import APIRouter
from typing import List

from .schemas import PingResponse, QueryRequest, QueryResponse, Source, RetrievalMeta
from ..retrieval import hybrid as hybrid_mod
from ..retrieval import rrf as rrf_mod

router = APIRouter()


@router.get("/ping", response_model=PingResponse)
async def ping():
    return {"status": "pong"}


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    # Placeholder retrieval flows: use hybrid retrieval as vector proxy.
    vector_candidates = hybrid_mod.hybrid_retrieve(req.query)
    # Represent as document dicts
    vect_docs = [
        {"id": f"v{i}", "score": max(0.0, 1.0 - i * 0.1), "content": c}
        for i, c in enumerate(vector_candidates[: req.top_k * 2])
    ]
    bm_docs = [{"id": f"b{i}", "score": max(0.0, 1.0 - i * 0.2), "content": req.query} for i in range(min(req.top_k, 1))]

    if req.retrieval_mode.startswith("hybrid"):
        fused = rrf_mod.reciprocal_rank_fusion([vect_docs, bm_docs], k=60)
    else:
        fused = vect_docs

    sources: List[Source] = [
        Source(document_id=d["id"], score=d.get("score", 0.0), content=d.get("content", "")) for d in fused[: req.top_k]
    ]

    retrieval = RetrievalMeta(
        mode=req.retrieval_mode,
        vector_candidates=len(vect_docs),
        bm25_candidates=len(bm_docs),
        rrf_candidates=len(fused),
        reranked_candidates=len(sources),
    )

    return QueryResponse(answer="(stub) This is a placeholder answer.", sources=sources, retrieval=retrieval)
