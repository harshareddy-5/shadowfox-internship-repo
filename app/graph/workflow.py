from typing import Dict, Any
from langgraph.graph import StateGraph, END
from app.core.config import settings
from app.graph.nodes import (
    validate_query_node,
    rewrite_query_node,
    retrieve_node,
    rerank_node,
    filter_context_node,
    generate_answer_node,
    groundedness_check_node,
    final_response_node
)
from app.graph.state import RAGGraphState


def should_retry_or_finish(state: RAGGraphState) -> str:
    """Conditional edge router following groundedness check."""
    is_grounded = state.get("grounded", True)
    retry_count = state.get("retry_count", 0)
    max_retries = settings.MAX_GROUNDEDNESS_RETRIES

    if is_grounded or retry_count >= max_retries:
        return "final_response"
    else:
        return "generate_answer"


def build_rag_workflow() -> StateGraph:
    """Construct and compile the multi-step LangGraph RAG workflow."""
    workflow = StateGraph(RAGGraphState)

    # Add nodes
    workflow.add_node("validate_query", validate_query_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("filter_context", filter_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("groundedness_check", groundedness_check_node)
    workflow.add_node("final_response", final_response_node)

    # Set entry point
    workflow.set_entry_point("validate_query")

    # Add linear edges
    workflow.add_edge("validate_query", "rewrite_query")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "filter_context")
    workflow.add_edge("filter_context", "generate_answer")
    workflow.add_edge("generate_answer", "groundedness_check")

    # Add conditional retry edge
    workflow.add_conditional_edges(
        "groundedness_check",
        should_retry_or_finish,
        {
            "final_response": "final_response",
            "generate_answer": "generate_answer"
        }
    )

    workflow.add_edge("final_response", END)

    return workflow.compile()


# Global compiled app graph
rag_graph_app = build_rag_workflow()
