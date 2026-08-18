from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str

    llm_model: str = "gpt-4.1-mini"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    vector_size: int = 384
    # If true the app will recreate the Qdrant collection when dimensions mismatch
    force_recreate_qdrant: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()