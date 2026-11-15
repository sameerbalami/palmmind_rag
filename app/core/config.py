from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / `.env`."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Palm Mind RAG API"

    # LLM (Google Gemini)
    gemini_api_key: str
    chat_model: str = "gemini-3.6-flash"
    # Tried in order when the primary model is overloaded (HTTP 429/503).
    chat_fallback_models: list[str] = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash"]
    llm_timeout_seconds: int = 60
    # Longest an upload waits in total for the embedding rate limit to reset before returning 429.
    llm_max_rate_limit_wait_seconds: int = 180
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768

    # Vector store (Qdrant)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents"

    # Chat memory (Redis)
    redis_url: str = "redis://localhost:6379/0"
    chat_history_max_messages: int = 20
    chat_history_ttl_seconds: int = 60 * 60 * 24

    # Metadata / bookings (SQL)
    database_url: str = "sqlite:///./app.db"

    # Ingestion & retrieval
    max_upload_mb: int = 10
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
