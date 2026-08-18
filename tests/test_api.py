from fastapi.testclient import TestClient
from app.main import app


def test_query_endpoint():
    client = TestClient(app)
    payload = {"query": "What is RAG?", "retrieval_mode": "hybrid_rerank", "top_k": 3}
    resp = client.post("/query", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "retrieval" in data
