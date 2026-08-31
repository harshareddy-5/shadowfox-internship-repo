import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import faiss
import numpy as np

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.core.logging import logger
from app.schemas.document import DocumentChunkMetadata, DocumentMetadata


class FAISSVectorStore:
    """
    FAISS-based vector index with metadata persistence and document-scoped filtering.
    Uses Inner Product (IndexFlatIP) which corresponds to Cosine Similarity for L2-normalized embeddings.
    """

    def __init__(
        self,
        dimension: int = None,
        index_dir: Path = None
    ):
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.index_dir = Path(index_dir or settings.INDEXES_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.index_dir / "faiss_index.bin"
        self.metadata_file = self.index_dir / "metadata.json"

        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.chunks_metadata: List[DocumentChunkMetadata] = []
        self.documents_metadata: Dict[str, DocumentMetadata] = {}

        # Load existing index from disk if present
        self.load_from_disk()

    def add_chunks(
        self,
        chunks: List[DocumentChunkMetadata],
        embeddings: np.ndarray,
        doc_metadata: DocumentMetadata
    ) -> None:
        """
        Add document chunks and their corresponding normalized embeddings to the FAISS index.
        """
        if len(chunks) == 0:
            return

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})."
            )

        try:
            # Ensure embeddings are float32 and contiguous
            embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
            self.index.add(embeddings_f32)

            self.chunks_metadata.extend(chunks)
            self.documents_metadata[doc_metadata.document_id] = doc_metadata

            self.save_to_disk()
            logger.info(
                f"Successfully added {len(chunks)} chunks for document '{doc_metadata.filename}' (ID: {doc_metadata.document_id}). Total vectors in index: {self.index.ntotal}."
            )
        except Exception as e:
            logger.error(f"Failed to add chunks to FAISS store: {str(e)}")
            raise VectorStoreError(f"Error adding vectors to index: {str(e)}")

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        selected_document_ids: Optional[List[str]] = None
    ) -> List[Tuple[DocumentChunkMetadata, float]]:
        """
        Perform similarity search in FAISS.
        Supports document-scoped filtering by checking `selected_document_ids`.
        Returns a list of (DocumentChunkMetadata, similarity_score) tuples sorted by score descending.
        """
        if self.index.ntotal == 0:
            return []

        # Clean selected_document_ids filter
        active_filter = set(selected_document_ids) if selected_document_ids else None

        # Fetch more candidates initially if filter is active
        fetch_k = min(self.index.ntotal, top_k * 5 if active_filter else top_k)

        query_f32 = np.ascontiguousarray(query_embedding, dtype=np.float32)
        scores, indices = self.index.search(query_f32, fetch_k)

        results: List[Tuple[DocumentChunkMetadata, float]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks_metadata):
                continue

            chunk = self.chunks_metadata[idx]

            # Apply document-scoped filtering
            if active_filter and chunk.document_id not in active_filter:
                continue

            results.append((chunk, float(score)))

            if len(results) >= top_k:
                break

        return results

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its associated chunks.
        Rebuilds the index with remaining chunks to maintain alignment.
        """
        if document_id not in self.documents_metadata:
            return False

        logger.info(f"Deleting document ID '{document_id}' from vector store...")

        # Filter out chunks for deleted document
        remaining_chunks = [c for c in self.chunks_metadata if c.document_id != document_id]
        del self.documents_metadata[document_id]

        if len(remaining_chunks) == len(self.chunks_metadata):
            return False

        # Reset index and rebuild from remaining chunks if any
        self.chunks_metadata = []
        self.index = faiss.IndexFlatIP(self.dimension)

        if remaining_chunks:
            # We need embedding service to re-embed remaining chunks
            from app.embeddings.embedding_service import get_embedding_service
            embedding_service = get_embedding_service()
            texts = [c.text for c in remaining_chunks]
            embeddings = embedding_service.embed_texts(texts)

            embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
            self.index.add(embeddings_f32)
            self.chunks_metadata = remaining_chunks

        self.save_to_disk()
        logger.info(f"Document '{document_id}' deleted. Index updated (Total chunks remaining: {self.index.ntotal}).")
        return True

    def get_indexed_documents(self) -> List[DocumentMetadata]:
        """Return list of all currently indexed document metadata."""
        return list(self.documents_metadata.values())

    def get_document_chunks(self, document_id: str) -> List[DocumentChunkMetadata]:
        """Get all chunks belonging to a specific document."""
        return [c for c in self.chunks_metadata if c.document_id == document_id]

    def save_to_disk(self) -> None:
        """Persist FAISS index and metadata store to disk."""
        try:
            faiss.write_index(self.index, str(self.index_file))

            metadata_payload = {
                "documents": {doc_id: doc.model_dump() for doc_id, doc in self.documents_metadata.items()},
                "chunks": [chunk.model_dump() for chunk in self.chunks_metadata]
            }

            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata_payload, f, indent=2)

            logger.debug(f"Saved FAISS index ({self.index.ntotal} vectors) and metadata to '{self.index_dir}'.")
        except Exception as e:
            logger.error(f"Error saving FAISS store to disk: {str(e)}")
            raise VectorStoreError(f"Failed to persist vector index: {str(e)}")

    def load_from_disk(self) -> None:
        """Load FAISS index and metadata store from disk if present."""
        if not self.index_file.exists() or not self.metadata_file.exists():
            logger.info("No existing FAISS index found on disk. Initializing new index.")
            return

        try:
            self.index = faiss.read_index(str(self.index_file))

            with open(self.metadata_file, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.documents_metadata = {
                doc_id: DocumentMetadata(**doc_data)
                for doc_id, doc_data in payload.get("documents", {}).items()
            }
            self.chunks_metadata = [
                DocumentChunkMetadata(**chunk_data)
                for chunk_data in payload.get("chunks", [])
            ]

            logger.info(
                f"Loaded existing FAISS index from disk. Total indexed vectors: {self.index.ntotal}, Total documents: {len(self.documents_metadata)}."
            )
        except Exception as e:
            logger.warning(f"Failed to load vector store from disk ({str(e)}). Re-initializing empty index.")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks_metadata = []
            self.documents_metadata = {}


# Global singleton vector store instance
_vector_store_instance: Optional[FAISSVectorStore] = None

def get_vector_store() -> FAISSVectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = FAISSVectorStore()
    return _vector_store_instance
