from pydantic import BaseSettings


class Settings(BaseSettings):
    env: str = "development"
    database_url: str = "sqlite:///./data.db"


settings = Settings()
