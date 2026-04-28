# FastAPI application entry point for the investment research assistant.
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.routes import router
import logging


# Basic console logging is enough for local debugging and benchmarking.
logging.basicConfig(level=logging.INFO)

# FastAPI app metadata is shown automatically in the OpenAPI docs.
app = FastAPI(
    title="Agentic RAG Investment Research Assistant",
    version="0.1.0",
    description="FastAPI + LangChain + Gemini API investment research assistant",
)

app.include_router(router)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
# Simple frontend for local demos and manual testing.
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
