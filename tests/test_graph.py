from app.graph.nodes import (
    validate_query_node,
    filter_context_node,
    generate_answer_node,
    groundedness_check_node,
    final_response_node
)
from app.graph.workflow import rag_graph_app
from app.schemas.query import RetrievedChunk


def test_validate_query_node():
    """Test query validation node."""
    res_valid = validate_query_node({"original_query": "  What is AI?  ", "stages_passed": []})
    assert res_valid["original_query"] == "What is AI?"

    res_empty = validate_query_node({"original_query": "   ", "stages_passed": []})
    assert res_empty.get("error_message") is not None


def test_context_filter_node():
    """Test context quality relevance threshold filtering."""
    chunks = [
        RetrievedChunk(chunk_id="c1", document_id="d1", filename="f.txt", page=1, chunk_index=0, text="Relevant", similarity_score=0.85, rerank_score=0.88),
        RetrievedChunk(chunk_id="c2", document_id="d1", filename="f.txt", page=1, chunk_index=1, text="Irrelevant", similarity_score=0.10, rerank_score=0.12)
    ]
    state = {"reranked_chunks": chunks, "stages_passed": []}
    res = filter_context_node(state)

    assert len(res["filtered_context"]) == 1
    assert res["filtered_context"][0].chunk_id == "c1"
    assert res["retrieval_quality"] == "HIGH"


def test_empty_context_refusal():
    """Test safe refusal answer generation when filtered context is empty."""
    state = {"filtered_context": [], "original_query": "What is Quantum Computing?", "stages_passed": []}
    res = generate_answer_node(state)

    assert "couldn't find sufficient information" in res["generated_answer"].lower()
    assert res["grounded"] is True
    assert res["groundedness_result"] == "INSUFFICIENT_INFO_RESPONSE"
