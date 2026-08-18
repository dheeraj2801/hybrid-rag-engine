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


Use async routes when your endpoint performs I/O-bound work (database calls, HTTP requests, long polling). Async handlers enable high concurrency with fewer threads by allowing the event loop to schedule other work while awaiting I/O. Prefer sync handlers for CPU-bound tasks or when using libraries that are not async-aware; offload CPU-heavy work to background workers or process pools.

Interactive docs: `/docs` (Swagger UI) and `/redoc` (ReDoc).

Health and metrics endpoints: expose `/health` and `/metrics` for monitoring and liveness/readiness checks.

Dependency injection: `Depends()` provides a way to declare and reuse components like DB sessions or auth validators.

BackgroundTasks: execute short, non-critical work after a response is returned.

Example — dependency:

```python
from fastapi import Depends

def get_db():
	db = SessionLocal()
	try:
		yield db
	finally:
		db.close()

@app.get('/items')
def list_items(db=Depends(get_db)):
	return db.query(Item).all()
```

Return structured error payloads with an `error` object and an appropriate HTTP status. Use Pydantic models to validate input and return `422` for invalid payloads. For authentication/authorization use `401`/`403` and include instructions for token renewal when appropriate.

Combine unit tests via `TestClient`, CI-based lint/type checks, and production readiness by running under a process manager and exposing `/metrics` for Prometheus scraping. This combined view ensures the same code paths are exercised in tests and production, improving reliability.
