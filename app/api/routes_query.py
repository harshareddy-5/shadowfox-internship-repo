import json
import time
from typing import AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.logging import logger
from app.graph.workflow import rag_graph_app
from app.schemas.query import QueryRequest
from app.schemas.response import RAGResponse

router = APIRouter(prefix="/query", tags=["RAG Knowledge Q&A Engine"])


@router.post(
    "",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute multi-step grounded RAG query"
)
async def query_rag_engine(request: QueryRequest):
    """
    Executes the full LangGraph RAG workflow:
    validate -> rewrite -> vector retrieve -> rerank -> context filter -> grounded answer gen -> groundedness check -> citations.
    """
    start_time = time.time()
    logger.info(f"Received RAG query request: '{request.query[:60]}...' (Selected docs: {request.selected_document_ids})")

    initial_state = {
        "original_query": request.query,
        "rewritten_query": None,
        "selected_document_ids": request.selected_document_ids,
        "top_k": request.top_k or settings.RETRIEVAL_TOP_K,
        "enable_reranking": request.enable_reranking if request.enable_reranking is not None else settings.ENABLE_RERANKER,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "filtered_context": [],
        "generated_answer": None,
        "citations": [],
        "retrieval_scores": [],
        "retrieval_quality": "INSUFFICIENT",
        "groundedness_result": "UNGROUNDED",
        "grounded": False,
        "retry_count": 0,
        "stages_passed": [],
        "error_message": None
    }

    try:
        final_state = rag_graph_app.invoke(initial_state)
        elapsed_time = round(time.time() - start_time, 3)

        return RAGResponse(
            original_query=final_state.get("original_query", request.query),
            rewritten_query=final_state.get("rewritten_query"),
            answer=final_state.get("generated_answer", "No response generated."),
            citations=final_state.get("citations", []),
            retrieved_chunks=final_state.get("filtered_context", []),
            retrieval_quality=final_state.get("retrieval_quality", "INSUFFICIENT"),
            grounded=final_state.get("grounded", False),
            selected_document_ids=request.selected_document_ids,
            execution_time_seconds=elapsed_time,
            stages_passed=final_state.get("stages_passed", [])
        )
    except Exception as e:
        logger.error(f"Error executing RAG workflow: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG execution workflow error: {str(e)}"
        )


@router.post(
    "/stream",
    summary="Execute RAG query with streaming token response"
)
async def query_rag_engine_stream(request: QueryRequest):
    """
    Execute RAG retrieval and stream the generated answer tokens using Server-Sent Events (SSE) / chunked transfer.
    """
    start_time = time.time()

    initial_state = {
        "original_query": request.query,
        "rewritten_query": None,
        "selected_document_ids": request.selected_document_ids,
        "top_k": request.top_k or settings.RETRIEVAL_TOP_K,
        "enable_reranking": request.enable_reranking if request.enable_reranking is not None else settings.ENABLE_RERANKER,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "filtered_context": [],
        "generated_answer": None,
        "citations": [],
        "retrieval_scores": [],
        "retrieval_quality": "INSUFFICIENT",
        "groundedness_result": "UNGROUNDED",
        "grounded": False,
        "retry_count": 0,
        "stages_passed": [],
        "error_message": None
    }

    async def event_generator() -> AsyncGenerator[str, None]:
        # Run graph workflow up to generation
        final_state = rag_graph_app.invoke(initial_state)

        # Emit metadata header event
        meta_event = {
            "type": "metadata",
            "original_query": final_state.get("original_query"),
            "rewritten_query": final_state.get("rewritten_query"),
            "retrieval_quality": final_state.get("retrieval_quality"),
            "grounded": final_state.get("grounded"),
            "citations": [c.model_dump() for c in final_state.get("citations", [])],
            "retrieved_chunks": [c.model_dump() for c in final_state.get("filtered_context", [])],
            "stages_passed": final_state.get("stages_passed", [])
        }
        yield f"data: {json.dumps(meta_event)}\n\n"

        # Stream answer text token by token or in chunks
        answer = final_state.get("generated_answer", "")
        chunk_size = 8
        words = answer.split(" ")
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i : i + chunk_size]) + " "
            token_event = {"type": "token", "content": chunk_text}
            yield f"data: {json.dumps(token_event)}\n\n"

        elapsed = round(time.time() - start_time, 3)
        end_event = {"type": "end", "execution_time_seconds": elapsed}
        yield f"data: {json.dumps(end_event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
