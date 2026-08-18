from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel

from app.graph.graph import rag_graph
from app.retrieval.vector_store import ensure_collection
from app.config.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure Qdrant collection exists before serving requests
    created = ensure_collection(settings.qdrant_collection, settings.vector_size)
    if created:
        print(f"Created Qdrant collection: {settings.qdrant_collection}")
    yield


app = FastAPI(title="Hybrid RAG Engine", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


@app.get("/ping")
def ping(request: Request):
    return "Hello"


@app.post("/query")
async def query(request: QueryRequest):
    result = rag_graph.invoke({"query": request.query})

    return {
        "answer": result["answer"],
        "sources": [
            {"content": document.page_content, "metadata": document.metadata}
            for document in result["documents"]
        ],
    }
