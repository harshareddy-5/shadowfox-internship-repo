"""
RAG Pipeline Orchestrator Module.

Coordinates document chunk retrieval, context formatting, vector distance checking,
and grounded answer generation via Groq LLM API.
"""

from typing import Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from src.retriever import search_relevant_chunks, format_context_from_chunks
from src.llm import generate_grounded_answer


NO_INFO_MSG = (
    "I couldn't find sufficient information in the "
    "uploaded documents to answer this question."
)


def run_rag_pipeline(
    vector_store: Optional[FAISS],
    question: str,
    top_k: int = 4,
    distance_threshold: float = 1.35,
    api_key: Optional[str] = None,
    model_name: str = "llama-3.3-70b-versatile"
) -> Dict[str, Any]:
    """
    Executes the end-to-end RAG workflow:
      Question
      -> Question Embedding
      -> Vector Similarity Search (FAISS)
      -> Relevant Document Chunks
      -> Context Construction
      -> Grounded LLM Generation (Groq)
      -> Answer + Source Metadata

    Args:
        vector_store (Optional[FAISS]): Processed FAISS vector store index.
        question (str): User natural language question.
        top_k (int): Number of chunks to retrieve. Default 4.
        distance_threshold (float): L2 distance cutoff threshold.
        api_key (Optional[str]): Groq API Key.
        model_name (str): Groq LLM model name.

    Returns:
        Dict[str, Any]: Payload containing success status, answer, sources, and errors.
    """
    # 1. Validate Question Input
    if not question or not question.strip():
        return {
            "success": False,
            "answer": "",
            "sources": [],
            "chunk_count": 0,
            "error": "Please enter a question."
        }

    # 2. Check Vector Store State
    if vector_store is None:
        return {
            "success": False,
            "answer": "",
            "sources": [],
            "chunk_count": 0,
            "error": "No document index found. Please upload and process documents first."
        }

    # 3. Vector Similarity Search
    retrieved_results = search_relevant_chunks(
        vector_store=vector_store,
        query=question,
        top_k=top_k,
        distance_threshold=distance_threshold
    )

    # 4. Check for low relevance / empty search results
    if not retrieved_results:
        return {
            "success": True,
            "answer": NO_INFO_MSG,
            "sources": [],
            "chunk_count": 0,
            "error": None
        }

    # 5. Format Context and Extract Source Metadata
    context_str = format_context_from_chunks(retrieved_results)

    sources = []
    for doc, score in retrieved_results:
        sources.append({
            "source": doc.metadata.get("source", "Unknown Document"),
            "page": doc.metadata.get("page"),
            "distance": round(score, 3),
            "content": doc.page_content
        })

    # 6. LLM Generation
    try:
        answer = generate_grounded_answer(
            context=context_str,
            question=question,
            api_key=api_key,
            model_name=model_name
        )
        return {
            "success": True,
            "answer": answer,
            "sources": sources,
            "chunk_count": len(sources),
            "error": None
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        return {
            "success": False,
            "answer": "",
            "sources": sources,
            "chunk_count": len(sources),
            "error": str(e)
        }
