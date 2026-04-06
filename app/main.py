# FastAPI application entry point for the investment research assistant.
from fastapi import FastAPI

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


@app.get("/")
# Friendly root route that points users to the generated API docs.
def root() -> dict:
    return {
        "name": app.title,
        "version": app.version,
        "docs": "/docs",
    }
