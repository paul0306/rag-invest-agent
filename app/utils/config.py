# Centralized runtime configuration loaded from environment variables.
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


# Application settings with sensible defaults for local development.
class Settings(BaseModel):
    google_api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", ""))
    model_name: str = Field(default_factory=lambda: os.getenv("MODEL_NAME", "gemini-2.5-flash"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview"))
    vector_store_path: Path = Field(default_factory=lambda: Path(os.getenv("VECTOR_STORE_PATH", "vector_store")))
    data_dir: Path = Field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    news_base_url: str = Field(default_factory=lambda: os.getenv("NEWS_BASE_URL", "https://news.google.com/rss/search"))
    news_max_results: int = Field(default_factory=lambda: int(os.getenv("NEWS_MAX_RESULTS", "6")))
    news_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("NEWS_TIMEOUT_SECONDS", "8")))
    retriever_k: int = 4
    mmr_fetch_k: int = 8
    chunk_size: int = 700
    chunk_overlap: int = 120
    temperature: float = 0.1


# Cache parsed settings so repeated imports do not re-read the environment.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
