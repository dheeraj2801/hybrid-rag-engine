# FastAPI — Practical Notes and Patterns

FastAPI is a modern, high-performance web framework for building APIs with Python 3.8+. It is async-first, uses Pydantic for data validation, and automatically generates OpenAPI docs.

Core features

- Type-annotated request/response models with Pydantic
- Automatic OpenAPI and interactive docs at `/docs` (Swagger UI) and `/redoc`
- Dependency injection for reusable components (DB sessions, clients)
- Background tasks, WebSockets, and Streaming responses

Simple example

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
	query: str

@app.post('/query')
async def handle_query(q: Query):
	# placeholder — integrate retrieval and LLM generation
	return {"answer": f"You asked: {q.query}", "sources": []}
```

Dependency injection

- Use `Depends(...)` to inject common resources such as database sessions, clients, or configuration objects.

Background tasks and streaming

- `BackgroundTasks` runs short-lived work after returning a response.
- Use `StreamingResponse` for large responses or server-sent events.

Testing

- Use `TestClient` from `starlette.testclient` / `fastapi.testclient` for synchronous tests.
- Example with `pytest`:

```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_ping():
	r = client.get('/ping')
	assert r.status_code == 200
```

Deployment

- Development: `uvicorn app.main:app --reload`
- Production: run under Gunicorn with Uvicorn workers, or use `uvicorn` with process manager. Example:
  ```bash
  gunicorn -k uvicorn.workers.UvicornWorker -w 4 app.main:app
  ```

Observability

- Use structured logging and attach request IDs via middleware.
- Export metrics with Prometheus client and expose `/metrics` for scraping.

Security

- Validate inputs strictly with Pydantic models.
- Rate-limit and authenticate endpoints (OAuth2, API keys, JWT).

When to use FastAPI

- Building async APIs with strong validation guarantees and automatic docs.
- Integrating with async database drivers and external async services.
