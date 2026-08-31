from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.query import Citation, RetrievedChunk


class RAGResponse(BaseModel):
    """Complete response returned by the DocuMind AI RAG system."""
    original_query: str
    rewritten_query: Optional[str] = None
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    retrieval_quality: str = Field(..., description="HIGH, MEDIUM, LOW, or INSUFFICIENT")
    grounded: bool = Field(..., description="True if answer is strictly grounded in retrieved context")
    selected_document_ids: Optional[List[str]] = None
    execution_time_seconds: float = 0.0
    stages_passed: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """System status and health check response."""
    status: str = "healthy"
    version: str
    vector_store_indexed_documents: int
    vector_store_total_chunks: int
    llm_provider: str
    embedding_model: str


class ErrorResponse(BaseModel):
    """Standardized API error message response."""
    error: str
    message: str
    status_code: int
