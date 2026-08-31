from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.core.logging import logger


class DocuMindException(Exception):
    """Base exception for DocuMind AI application errors."""

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class DocumentValidationError(DocuMindException):
    """Raised when an uploaded document fails validation."""
    def __init__(self, message: str):
        super().__init__(message, status_code=status.HTTP_400_BAD_REQUEST)


class DocumentProcessingError(DocuMindException):
    """Raised when document parsing or chunking fails."""
    def __init__(self, message: str):
        super().__init__(message, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


class VectorStoreError(DocuMindException):
    """Raised when FAISS indexing or retrieval operations fail."""
    def __init__(self, message: str):
        super().__init__(message, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


class LLMServiceError(DocuMindException):
    """Raised when the LLM provider fails to generate a response."""
    def __init__(self, message: str):
        super().__init__(message, status_code=status.HTTP_502_BAD_GATEWAY)


class InvalidQueryError(DocuMindException):
    """Raised when query validation fails."""
    def __init__(self, message: str):
        super().__init__(message, status_code=status.HTTP_400_BAD_REQUEST)


class DocumentNotFoundError(DocuMindException):
    """Raised when a requested document ID does not exist."""
    def __init__(self, message: str = "Requested document was not found."):
        super().__init__(message, status_code=status.HTTP_404_NOT_FOUND)


async def documind_exception_handler(request: Request, exc: DocuMindException) -> JSONResponse:
    """FastAPI exception handler for custom DocuMind exceptions."""
    logger.error(f"Application exception on {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "status_code": exc.status_code
        }
    )
