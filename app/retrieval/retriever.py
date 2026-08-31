from typing import List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.embeddings.embedding_service import get_embedding_service, EmbeddingService
from app.retrieval.faiss_store import get_vector_store, FAISSVectorStore
from app.schemas.query import RetrievedChunk


class DocumentRetriever:
    """
    Executes document-scoped vector retrieval against the FAISS store.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService = None,
        vector_store: FAISSVectorStore = None
    ):
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        selected_document_ids: Optional[List[str]] = None
    ) -> List[RetrievedChunk]:
        """
        Embed query and search vector store with optional document-scoped filtering.
        Returns list of RetrievedChunk objects.
        """
        top_k = top_k or settings.RETRIEVAL_TOP_K
        query_embedding = self.embedding_service.embed_query(query)

        raw_results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            selected_document_ids=selected_document_ids
        )

        retrieved_chunks = []
        for chunk_meta, score in raw_results:
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_meta.chunk_id,
                    document_id=chunk_meta.document_id,
                    filename=chunk_meta.filename,
                    page=chunk_meta.page,
                    chunk_index=chunk_meta.chunk_index,
                    text=chunk_meta.text,
                    similarity_score=round(score, 4)
                )
            )

        logger.info(
            f"Retrieved {len(retrieved_chunks)} candidate chunks for query '{query[:50]}...' (Document filter: {selected_document_ids or 'All'})."
        )
        return retrieved_chunks


def get_retriever() -> DocumentRetriever:
    return DocumentRetriever()
