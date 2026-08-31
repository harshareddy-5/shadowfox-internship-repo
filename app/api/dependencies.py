from app.services.document_service import get_document_service, DocumentService
from app.embeddings.embedding_service import get_embedding_service, EmbeddingService
from app.retrieval.faiss_store import get_vector_store, FAISSVectorStore
from app.retrieval.retriever import get_retriever, DocumentRetriever
from app.retrieval.reranker import get_reranker_service, RerankerService
from app.generation.llm import get_llm_provider, BaseLLMProvider


def get_doc_service_dep() -> DocumentService:
    return get_document_service()


def get_retriever_dep() -> DocumentRetriever:
    return get_retriever()


def get_llm_dep() -> BaseLLMProvider:
    return get_llm_provider()
