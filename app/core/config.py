import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and configuration management."""

    # Project metadata
    PROJECT_NAME: str = "DocuMind AI — Production RAG Knowledge Assistant"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # API Keys
    GEMINI_API_KEY: str = ""

    # LLM Settings
    LLM_PROVIDER: str = "gemini"
    LLM_MODEL_NAME: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024

    # Embedding Settings
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384

    # Reranker Settings
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    ENABLE_RERANKER: bool = True
    RERANK_TOP_K: int = 4

    # Document Chunking Settings
    DEFAULT_CHUNK_SIZE: int = 800
    DEFAULT_CHUNK_OVERLAP: int = 120
    MAX_FILE_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".txt", ".md"]

    # Retrieval & Filtering Settings
    RETRIEVAL_TOP_K: int = 10
    RELEVANCE_THRESHOLD: float = 0.35
    MAX_GROUNDEDNESS_RETRIES: int = 2

    # File Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOADS_DIR: Path = BASE_DIR / "data" / "uploads"
    INDEXES_DIR: Path = BASE_DIR / "data" / "indexes"

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_directories(self) -> None:
        """Ensure required data directories exist."""
        self.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.INDEXES_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
