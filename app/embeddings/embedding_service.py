import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.core.logging import logger


class EmbeddingService:
    """
    Dedicated embedding service utilizing BAAI/bge-small-en-v1.5.
    Provides cached model loading, batch embedding generation, and L2 normalization.
    """

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION

    def _get_model(self) -> SentenceTransformer:
        """Lazy load and cache the SentenceTransformer model instance."""
        if EmbeddingService._model is None:
            logger.info(f"Loading embedding model '{self.model_name}'...")
            try:
                EmbeddingService._model = SentenceTransformer(self.model_name)
                logger.info(f"Embedding model '{self.model_name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{self.model_name}': {str(e)}")
                raise VectorStoreError(f"Embedding model initialization failed: {str(e)}")
        return EmbeddingService._model

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate L2-normalized float32 embeddings for a list of text strings.
        Returns numpy array of shape (N, dimension).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        try:
            model = self._get_model()
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error generating embeddings for {len(texts)} texts: {str(e)}")
            raise VectorStoreError(f"Embedding generation failed: {str(e)}")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate L2-normalized float32 embedding for a single query string.
        Returns numpy array of shape (1, dimension).
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty for embedding generation.")

        # Note: BGE models recommend query prefixing if applicable, but bge-small-en-v1.5 works directly
        return self.embed_texts([query])


# Global singleton instance provider
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
