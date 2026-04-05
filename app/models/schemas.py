from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Investment research question")


class AnalyzeResponse(BaseModel):
    result: str


class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
