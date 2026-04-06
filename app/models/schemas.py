# Pydantic schemas shared by the API layer.
from pydantic import BaseModel, Field


# Request schema for the /analyze endpoint.
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Investment research question")


# Response schema returned by the analysis endpoint.
class AnalyzeResponse(BaseModel):
    result: str


# Response schema returned by the health endpoint.
class HealthResponse(BaseModel):
    status: str
    vector_store_ready: bool
