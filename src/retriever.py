"""
Retriever Module.

Performs vector similarity search against the FAISS index with distance score thresholding,
and formats retrieved chunks into structured context prompts.
"""

from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def search_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = 4,
    distance_threshold: float = 1.35
) -> List[Tuple[Document, float]]:
    """
    Performs similarity search against the FAISS vector index using the query embedding.
    Returns the top-K relevant document chunks along with their distance scores.

    Distance Metrics (L2 Distance with Normalized Embeddings):
    - 0.0 to 0.8: High similarity / Strong match
    - 0.8 to 1.2: Moderate similarity
    - > 1.35: Low similarity / Irrelevant match

    Args:
        vector_store (FAISS): Loaded FAISS vector store.
        query (str): User natural language question.
        top_k (int): Number of top chunks to retrieve. Default 4.
        distance_threshold (float): Maximum distance threshold to consider a chunk relevant.

    Returns:
        List[Tuple[Document, float]]: List of (Document chunk, L2 distance score) tuples.
    """
    if not query or not query.strip():
        return []

    # Perform similarity search with score (L2 distance)
    results = vector_store.similarity_search_with_score(query.strip(), k=top_k)

    # Filter and format results
    filtered_results = []
    for doc, score in results:
        # Distance score in FAISS L2: lower is more similar
        if score <= distance_threshold:
            filtered_results.append((doc, float(score)))

    return filtered_results


def format_context_from_chunks(results: List[Tuple[Document, float]]) -> str:
    """
    Formats retrieved document chunks into a structured context string for the LLM prompt.

    Args:
        results (List[Tuple[Document, float]]): Retrieved document chunks and distance scores.

    Returns:
        str: Formatted context block with source and page metadata.
    """
    if not results:
        return ""

    context_blocks = []
    for i, (doc, _score) in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page")
        page_str = f"Page {page}" if page is not None else "Page N/A"

        block = (
            f"[DOCUMENT CHUNK {i}]\n"
            f"Source: {source} | {page_str}\n"
            f"Content:\n{doc.page_content}"
        )
        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)
