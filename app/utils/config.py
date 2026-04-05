import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Settings(BaseModel):
    google_api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", ""))
    model_name: str = Field(default_factory=lambda: os.getenv("MODEL_NAME", "gemini-2.5-flash"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview"))
    vector_store_path: Path = Field(default_factory=lambda: Path(os.getenv("VECTOR_STORE_PATH", "vector_store")))
    data_dir: Path = Field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    retriever_k: int = 4
    mmr_fetch_k: int = 8
    chunk_size: int = 700
    chunk_overlap: int = 120
    temperature: float = 0.1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
