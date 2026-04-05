from fastapi import APIRouter, HTTPException

from app.models.schemas import AnalyzeResponse, HealthResponse, QueryRequest
from app.services.agent_service import run_analysis
from app.services.rag_service import vector_store_ready

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", vector_store_ready=vector_store_ready())


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_stock(request: QueryRequest) -> AnalyzeResponse:
    try:
        result = run_analysis(request.query)
    except Exception as exc:  # pragma: no cover - defensive API wrapper
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AnalyzeResponse(result=result)
