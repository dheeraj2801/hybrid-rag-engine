from fastapi import FastAPI

def create_app() -> FastAPI:
    app = FastAPI(title="Hybrid RAG Benchmark")
    from .api.routes import router
    app.include_router(router)
    return app

app = create_app()
