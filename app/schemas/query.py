from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """API request for RAG Q&A."""
    query: str = Field(..., min_length=1, max_length=2000, description="User question or prompt")
    selected_document_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of document IDs to scope retrieval. If empty, searches all documents."
    )
    top_k: Optional[int] = Field(default=10, ge=1, le=50, description="Top K initial chunks to retrieve")
    enable_reranking: Optional[bool] = Field(default=True, description="Whether to apply neural reranking")

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query string cannot be empty or whitespace only.")
        return cleaned


class Citation(BaseModel):
    """Source reference for a generated answer."""
    document_id: str
    filename: str
    page: Optional[int] = None
    chunk_id: str
    relevance_score: float


class RetrievedChunk(BaseModel):
    """Details of a chunk retrieved during RAG execution."""
    chunk_id: str
    document_id: str
    filename: str
    page: Optional[int] = None
    chunk_index: int
    text: str
    similarity_score: float
    rerank_score: Optional[float] = None
