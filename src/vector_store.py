"""
Vector Store Module.

Wraps local FAISS vector indexing operations, including index creation,
serialization, and loading from disk.
"""

import os
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


def build_vector_store(chunks: List[Document], embedding_model: Any) -> FAISS:
    """
    Builds an in-memory FAISS vector index from document chunks and an embedding model.

    Args:
        chunks (List[Document]): List of chunked Document objects.
        embedding_model (Any): Embeddings generator instance.

    Returns:
        FAISS: Built FAISS vector store instance.

    Raises:
        ValueError: If chunks list is empty.
    """
    if not chunks:
        raise ValueError("Cannot build FAISS vector store with empty document chunks.")

    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store


def save_vector_store(vector_store: FAISS, index_path: str = "faiss_index") -> None:
    """
    Saves the FAISS index and docstore locally to disk.

    Args:
        vector_store (FAISS): FAISS vector store instance.
        index_path (str): Path to save index directory.
    """
    os.makedirs(index_path, exist_ok=True)
    vector_store.save_local(index_path)


def load_vector_store(index_path: str, embedding_model: Any) -> Optional[FAISS]:
    """
    Loads a saved FAISS index from disk.

    Args:
        index_path (str): Directory where index is saved.
        embedding_model (Any): Embeddings generator instance.

    Returns:
        Optional[FAISS]: Loaded FAISS vector store, or None if path does not exist.
    """
    if not os.path.exists(index_path):
        return None
    try:
        vector_store = FAISS.load_local(
            index_path,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error loading FAISS vector store: {e}")
        return None
