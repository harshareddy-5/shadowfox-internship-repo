import os
from pathlib import Path
from typing import List, Optional, Tuple
from fastapi import UploadFile

from app.core.config import settings
from app.core.exceptions import DocumentNotFoundError, DocumentProcessingError, DocumentValidationError
from app.core.logging import logger
from app.embeddings.embedding_service import get_embedding_service, EmbeddingService
from app.ingestion.chunker import DocumentChunker
from app.ingestion.loaders import DocumentLoader
from app.retrieval.faiss_store import get_vector_store, FAISSVectorStore
from app.schemas.document import DocumentChunkMetadata, DocumentMetadata, DocumentUploadResponse


class DocumentService:
    """
    Coordinates document lifecycle: upload, ingestion, chunking, embedding, FAISS indexing, metadata persistence, and deletion.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore = None,
        embedding_service: EmbeddingService = None,
        uploads_dir: Path = None
    ):
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_service or get_embedding_service()
        self.uploads_dir = Path(uploads_dir or settings.UPLOADS_DIR)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    async def upload_and_process_document(
        self,
        file: UploadFile,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> DocumentUploadResponse:
        """
        Process uploaded file: validate, save raw file, extract text, chunk, embed, and index in FAISS.
        """
        filename = file.filename or "uploaded_file.txt"
        file_bytes = await file.read()

        logger.info(f"Processing uploaded file '{filename}' ({len(file_bytes)} bytes)...")

        # Load & extract text
        doc_id, page_records, meta_dict = DocumentLoader.load_document(filename, file_bytes)

        # Chunk document
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks: List[DocumentChunkMetadata] = chunker.chunk_document(
            document_id=doc_id,
            filename=filename,
            page_records=page_records
        )

        if not chunks:
            raise DocumentProcessingError(f"Document '{filename}' could not be chunked properly.")

        # Save raw file to disk for archive/audit
        safe_filename = f"{doc_id}_{Path(filename).name}"
        saved_file_path = self.uploads_dir / safe_filename
        with open(saved_file_path, "wb") as f:
            f.write(file_bytes)

        # Build DocumentMetadata Pydantic object
        doc_metadata = DocumentMetadata(
            document_id=doc_id,
            filename=filename,
            file_type=meta_dict["file_type"],
            file_size_bytes=meta_dict["file_size_bytes"],
            total_pages=meta_dict["total_pages"],
            total_chunks=len(chunks)
        )

        # Generate embeddings
        chunk_texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts)

        # Index in FAISS
        self.vector_store.add_chunks(chunks=chunks, embeddings=embeddings, doc_metadata=doc_metadata)

        logger.info(f"Document '{filename}' (ID: {doc_id}) successfully indexed into FAISS with {len(chunks)} chunks.")

        return DocumentUploadResponse(
            document_id=doc_id,
            filename=filename,
            file_type=meta_dict["file_type"],
            file_size_bytes=meta_dict["file_size_bytes"],
            total_chunks=len(chunks),
            message="Document successfully processed, embedded, and indexed."
        )

    def list_documents(self) -> List[DocumentMetadata]:
        """Return list of all indexed documents."""
        return self.vector_store.get_indexed_documents()

    def get_document(self, document_id: str) -> DocumentMetadata:
        """Get metadata for a specific document ID."""
        docs = {d.document_id: d for d in self.vector_store.get_indexed_documents()}
        if document_id not in docs:
            raise DocumentNotFoundError(f"Document with ID '{document_id}' not found.")
        return docs[document_id]

    def get_document_chunks(self, document_id: str) -> List[DocumentChunkMetadata]:
        """Get all chunks for a specific document ID."""
        # Ensure document exists first
        self.get_document(document_id)
        return self.vector_store.get_document_chunks(document_id)

    def delete_document(self, document_id: str) -> bool:
        """Delete document from vector store and filesystem archive."""
        doc_meta = self.get_document(document_id)
        deleted = self.vector_store.delete_document(document_id)

        # Clean up archived file on disk
        if deleted:
            safe_filename = f"{document_id}_{Path(doc_meta.filename).name}"
            archived_file = self.uploads_dir / safe_filename
            if archived_file.exists():
                try:
                    os.remove(archived_file)
                except Exception as e:
                    logger.warning(f"Could not remove archived file '{archived_file}': {str(e)}")

        return deleted


def get_document_service() -> DocumentService:
    return DocumentService()
