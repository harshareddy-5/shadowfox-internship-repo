from typing import List, Tuple
from sentence_transformers import CrossEncoder

from app.core.config import settings
from app.core.logging import logger
from app.schemas.query import RetrievedChunk


class RerankerService:
    """
    Reranking service utilizing BAAI/bge-reranker-base (CrossEncoder).
    Provides graceful fallback to similarity ranking if model fails or is disabled.
    """

    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RerankerService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.model_name = settings.RERANKER_MODEL_NAME
        self.enabled = settings.ENABLE_RERANKER

    def _get_model(self) -> CrossEncoder | None:
        """Lazy load CrossEncoder model instance."""
        if not self.enabled:
            return None

        if RerankerService._model is None:
            logger.info(f"Loading reranker model '{self.model_name}'...")
            try:
                RerankerService._model = CrossEncoder(self.model_name)
                logger.info(f"Reranker model '{self.model_name}' loaded successfully.")
            except Exception as e:
                logger.warning(
                    f"Failed to load reranker model '{self.model_name}' ({str(e)}). Reranking will fallback to vector similarity."
                )
                RerankerService._model = None

        return RerankerService._model

    def rerank(
        self,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        top_n: int = None
    ) -> List[RetrievedChunk]:
        """
        Rerank retrieved chunks using CrossEncoder.
        Returns top_n reranked chunks with `rerank_score` field set.
        """
        top_n = top_n or settings.RERANK_TOP_K
        if not retrieved_chunks:
            return []

        model = self._get_model()

        if model is None:
            logger.info("Reranker unavailable or disabled. Using vector similarity ranking.")
            sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.similarity_score, reverse=True)
            return sorted_chunks[:top_n]

        try:
            pairs = [[query, chunk.text] for chunk in retrieved_chunks]
            scores = model.predict(pairs)

            for idx, score in enumerate(scores):
                # Sigmoid normalization for raw logit scores if needed
                norm_score = float(score)
                retrieved_chunks[idx].rerank_score = norm_score

            # Sort by rerank score descending
            reranked = sorted(retrieved_chunks, key=lambda x: x.rerank_score or -999.0, reverse=True)
            logger.info(f"Reranked {len(retrieved_chunks)} chunks to top {top_n}.")
            return reranked[:top_n]

        except Exception as e:
            logger.warning(f"Error during reranking execution ({str(e)}). Falling back to similarity scores.")
            sorted_chunks = sorted(retrieved_chunks, key=lambda x: x.similarity_score, reverse=True)
            return sorted_chunks[:top_n]


def get_reranker_service() -> RerankerService:
    return RerankerService()
