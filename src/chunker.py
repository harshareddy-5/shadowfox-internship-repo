"""
Document Chunker Module.

Splits extracted document texts into overlapping chunks using
RecursiveCharacterTextSplitter, preserving metadata and computing summary stats.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Document]:
    """
    Splits long document texts into smaller, overlapping chunks while preserving metadata.

    Why Chunking is Necessary:
    1. Context Window Limits: LLMs perform best with focused context.
    2. Precision Retrieval: Smaller chunks produce more specific vector embeddings.
    3. Overlap Utility: Overlapping chunks prevent losing context across boundaries.

    Args:
        documents (List[Document]): List of extracted page/file documents.
        chunk_size (int): Target character count per chunk. Default ~800.
        chunk_overlap (int): Overlap character count between chunks. Default ~150.

    Returns:
        List[Document]: List of chunked Document objects with preserved metadata.
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    # Ensure metadata integrity across all chunks
    for i, chunk in enumerate(chunks):
        if "chunk_id" not in chunk.metadata:
            chunk.metadata["chunk_id"] = i

    return chunks


def get_chunk_statistics(chunks: List[Document]) -> Dict[str, Any]:
    """
    Calculates summary statistics for a set of document chunks.

    Args:
        chunks (List[Document]): List of chunked Document objects.

    Returns:
        Dict[str, Any]: Statistics including chunk count, total characters, and avg chunk size.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "total_characters": 0,
            "avg_chunk_size": 0
        }

    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    avg_chars = round(total_chars / len(chunks), 1)

    return {
        "total_chunks": len(chunks),
        "total_characters": total_chars,
        "avg_chunk_size": avg_chars
    }
