from typing import TypedDict, List, Optional, Any, Dict
from app.schemas.query import Citation, RetrievedChunk


class RAGGraphState(TypedDict):
    """
    State object passed between LangGraph nodes during RAG execution.
    """
    original_query: str
    rewritten_query: Optional[str]
    selected_document_ids: Optional[List[str]]
    top_k: int
    enable_reranking: bool

    retrieved_chunks: List[RetrievedChunk]
    reranked_chunks: List[RetrievedChunk]
    filtered_context: List[RetrievedChunk]

    generated_answer: Optional[str]
    citations: List[Citation]
    retrieval_scores: List[float]
    retrieval_quality: str  # HIGH, MEDIUM, LOW, INSUFFICIENT
    groundedness_result: str  # GROUNDED, UNGROUNDED, INSUFFICIENT_INFO_RESPONSE
    grounded: bool

    retry_count: int
    stages_passed: List[str]
    error_message: Optional[str]
