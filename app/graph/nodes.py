import re
from typing import Dict, Any, List
from app.core.config import settings
from app.core.logging import logger
from app.generation.llm import get_llm_provider
from app.generation.prompts import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    QUERY_REWRITE_USER_PROMPT,
    GROUNDED_GENERATION_SYSTEM_PROMPT,
    GROUNDED_GENERATION_USER_PROMPT,
    GROUNDEDNESS_CHECK_SYSTEM_PROMPT,
    GROUNDEDNESS_CHECK_USER_PROMPT,
)
from app.graph.state import RAGGraphState
from app.retrieval.reranker import get_reranker_service
from app.retrieval.retriever import get_retriever
from app.schemas.query import Citation, RetrievedChunk


def validate_query_node(state: RAGGraphState) -> Dict[str, Any]:
    """Node 1: Validate input query."""
    query = state.get("original_query", "").strip()
    stages = list(state.get("stages_passed", []))
    stages.append("validate_query")

    if not query:
        return {
            "error_message": "Query cannot be empty.",
            "stages_passed": stages,
            "grounded": False,
            "retrieval_quality": "INSUFFICIENT"
        }

    return {
        "original_query": query,
        "stages_passed": stages
    }


def rewrite_query_node(state: RAGGraphState) -> Dict[str, Any]:
    """Node 2: Rewrite query for improved vector search if applicable."""
    stages = list(state.get("stages_passed", []))
    stages.append("rewrite_query")

    query = state.get("original_query", "")
    
    # Don't rewrite simple clear queries unnecessarily
    if len(query.split()) < 4 or "?" not in query:
        return {"rewritten_query": query, "stages_passed": stages}

    try:
        llm = get_llm_provider()
        prompt = QUERY_REWRITE_USER_PROMPT.format(query=query)
        rewritten = llm.generate(prompt=prompt, system_instruction=QUERY_REWRITE_SYSTEM_PROMPT)
        
        cleaned_rewritten = rewritten.strip('"\' \n')
        if cleaned_rewritten and len(cleaned_rewritten) > 3:
            logger.info(f"Query rewritten: '{query}' -> '{cleaned_rewritten}'")
            return {"rewritten_query": cleaned_rewritten, "stages_passed": stages}
    except Exception as e:
        logger.warning(f"Query rewriting skipped/failed ({str(e)}). Using original query.")

    return {"rewritten_query": query, "stages_passed": stages}


def retrieve_node(state: RAGGraphState) -> Dict[str, Any]:
    """Node 3: Execute vector similarity search against FAISS store."""
    stages = list(state.get("stages_passed", []))
    stages.append("retrieve")

    search_query = state.get("rewritten_query") or state.get("original_query")
    selected_docs = state.get("selected_document_ids")
    top_k = state.get("top_k", settings.RETRIEVAL_TOP_K)

    retriever = get_retriever()
    retrieved_chunks = retriever.retrieve(
        query=search_query,
        top_k=top_k,
        selected_document_ids=selected_docs
    )

    return {
        "retrieved_chunks": retrieved_chunks,
        "stages_passed": stages
    }


def rerank_node(state: RAGGraphState) -> Dict[str, Any]:
    """Node 4: Rerank retrieved candidates using CrossEncoder."""
    stages = list(state.get("stages_passed", []))
    stages.append("rerank")

    retrieved = state.get("retrieved_chunks", [])
    enable_rerank = state.get("enable_reranking", settings.ENABLE_RERANKER)
    search_query = state.get("rewritten_query") or state.get("original_query")

    if not retrieved or not enable_rerank:
        return {
            "reranked_chunks": retrieved[:settings.RERANK_TOP_K],
            "stages_passed": stages
        }

    reranker = get_reranker_service()
    reranked = reranker.rerank(query=search_query, retrieved_chunks=retrieved, top_n=settings.RERANK_TOP_K)

    return {
        "reranked_chunks": reranked,
        "stages_passed": stages
    }


def filter_context_node(state: RAGGraphState) -> Dict[str, Any]:
    """Node 5: Context Quality Filter - evaluates chunk relevance scores against threshold."""
    stages = list(state.get("stages_passed", []))
    stages.append("filter_context")

    reranked = state.get("reranked_chunks", [])
    threshold = settings.RELEVANCE_THRESHOLD

    filtered = []
    scores = []

    for chunk in reranked:
        score = chunk.rerank_score if chunk.rerank_score is not None else chunk.similarity_score
        scores.append(round(score, 4))
        if score >= threshold:
            filtered.append(chunk)

    # Determine retrieval quality rating
    if not filtered:
        quality = "INSUFFICIENT"
    elif max(scores, default=0.0) >= 0.70:
        quality = "HIGH"
    elif max(scores, default=0.0) >= 0.50:
        quality = "MEDIUM"
    else:
        quality = "LOW"

    logger.info(f"Context Quality Filter: {len(filtered)}/{len(reranked)} chunks passed threshold {threshold}. Quality: {quality}.")

    return {
        "filtered_context": filtered,
        "retrieval_scores": scores,
        "retrieval_quality": quality,
        "stages_passed": stages
    }


def generate_answer_node(state: RAGGraphState) -> Dict[str, Any]:
    """Node 6: Generate grounded answer using Gemini LLM."""
    stages = list(state.get("stages_passed", []))
    stages.append("generate_answer")

    context_chunks = state.get("filtered_context", [])
    query = state.get("original_query")
    retry_count = state.get("retry_count", 0)

    # If context is empty/insufficient, return refusal safely
    if not context_chunks:
        return {
            "generated_answer": "I couldn't find sufficient information about this in the uploaded documents.",
            "citations": [],
            "grounded": True,
            "groundedness_result": "INSUFFICIENT_INFO_RESPONSE",
            "stages_passed": stages
        }

    # Format context blocks with source metadata
    context_blocks = []
    for idx, chunk in enumerate(context_chunks, start=1):
        page_str = f", Page {chunk.page}" if chunk.page else ""
        context_blocks.append(
            f"--- Snippet {idx} [Document: {chunk.filename}{page_str}, Chunk: {chunk.chunk_index}] ---\n{chunk.text}"
        )

    formatted_context = "\n\n".join(context_blocks)

    try:
        llm = get_llm_provider()
        prompt = GROUNDED_GENERATION_USER_PROMPT.format(
            context_text=formatted_context,
            query=query
        )

        sys_prompt = GROUNDED_GENERATION_SYSTEM_PROMPT
        if retry_count > 0:
            sys_prompt += "\nWARNING: PREVIOUS GENERATION WAS UNGROUNDED. BE EXTREMELY STRICT AND USE ONLY EXPLICIT FACTS FROM CONTEXT."

        answer = llm.generate(prompt=prompt, system_instruction=sys_prompt)

        return {
            "generated_answer": answer,
            "stages_passed": stages
        }
    except Exception as e:
        logger.error(f"Answer generation error: {str(e)}")
        return {
            "generated_answer": f"An error occurred while generating answer: {str(e)}",
            "grounded": False,
            "groundedness_result": "UNGROUNDED",
            "stages_passed": stages
        }


def groundedness_check_node(state: RAGGraphState) -> Dict[str, Any]:
    """Node 7: Groundedness & Answer Validation Step."""
    stages = list(state.get("stages_passed", []))
    stages.append("groundedness_check")

    answer = state.get("generated_answer", "")
    context_chunks = state.get("filtered_context", [])
    retry_count = state.get("retry_count", 0)

    # Refusal responses are inherently grounded
    if "couldn't find sufficient information" in answer.lower() or not context_chunks:
        return {
            "grounded": True,
            "groundedness_result": "INSUFFICIENT_INFO_RESPONSE",
            "stages_passed": stages
        }

    context_text = "\n".join([c.text for c in context_chunks])

    try:
        llm = get_llm_provider()
        eval_prompt = GROUNDEDNESS_CHECK_USER_PROMPT.format(
            context_text=context_text,
            generated_answer=answer
        )
        eval_res = llm.generate(prompt=eval_prompt, system_instruction=GROUNDEDNESS_CHECK_SYSTEM_PROMPT)

        first_line = eval_res.split("\n")[0].strip().upper()
        if "GROUNDED" in first_line:
            is_grounded = True
            result = "GROUNDED"
        elif "INSUFFICIENT" in first_line:
            is_grounded = True
            result = "INSUFFICIENT_INFO_RESPONSE"
        else:
            is_grounded = False
            result = "UNGROUNDED"

        logger.info(f"Groundedness verification result: {result} (Retry count: {retry_count}).")

        return {
            "grounded": is_grounded,
            "groundedness_result": result,
            "retry_count": retry_count + (0 if is_grounded else 1),
            "stages_passed": stages
        }
    except Exception as e:
        logger.warning(f"Groundedness check failed/skipped ({str(e)}). Assuming grounded.")
        return {
            "grounded": True,
            "groundedness_result": "GROUNDED",
            "stages_passed": stages
        }


def final_response_node(state: RAGGraphState) -> Dict[str, Any]:
    """Node 8: Build final citations and assemble RAG response state."""
    stages = list(state.get("stages_passed", []))
    stages.append("final_response")

    context_chunks = state.get("filtered_context", [])
    grounded = state.get("grounded", True)
    answer = state.get("generated_answer", "")

    # Build citations from filtered context chunks
    citations = []
    seen = set()
    for chunk in context_chunks:
        cit_key = (chunk.document_id, chunk.page, chunk.chunk_id)
        if cit_key not in seen:
            seen.add(cit_key)
            citations.append(
                Citation(
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page=chunk.page,
                    chunk_id=chunk.chunk_id,
                    relevance_score=chunk.rerank_score or chunk.similarity_score
                )
            )

    # If ungrounded after max retries, return safe refusal fallback
    if not grounded:
        answer = "I'm sorry, but a strictly grounded answer could not be generated from the retrieved context."

    return {
        "generated_answer": answer,
        "citations": citations,
        "stages_passed": stages
    }
