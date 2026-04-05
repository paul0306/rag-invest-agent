from fastapi import FastAPI

from app.api.routes import router
import logging


logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Agentic RAG Investment Research Assistant",
    version="0.1.0",
    description="FastAPI + LangChain + Gemini API investment research assistant",
)

app.include_router(router)


@app.get("/")
def root() -> dict:
    return {
        "name": app.title,
        "version": app.version,
        "docs": "/docs",
    }
