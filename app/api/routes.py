# API routes for health checks and stock-analysis requests.
from fastapi import APIRouter, HTTPException

from app.models.schemas import AnalyzeResponse, HealthResponse, QueryRequest
from app.services.agent_service import run_analysis
from app.services.rag_service import vector_store_ready

# Single router keeps the FastAPI surface area intentionally small.
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
# Basic health endpoint used by tests or deployment checks.
def health() -> HealthResponse:
    return HealthResponse(status="ok", vector_store_ready=vector_store_ready())


@router.post("/analyze", response_model=AnalyzeResponse)
# Main inference endpoint that wraps the agent with simple HTTP error handling.
async def analyze_stock(request: QueryRequest) -> AnalyzeResponse:
    try:
        result = run_analysis(request.query)
    except Exception as exc:  # pragma: no cover - defensive API wrapper
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AnalyzeResponse(result=result)
