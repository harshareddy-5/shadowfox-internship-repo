from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata representing an uploaded and processed document."""
    document_id: str = Field(..., description="Unique ID for the document")
    filename: str = Field(..., description="Original name of the uploaded file")
    file_type: str = Field(..., description="Extension/MIME type (.pdf, .txt, .md)")
    file_size_bytes: int = Field(..., description="File size in bytes")
    total_pages: Optional[int] = Field(default=None, description="Total pages if PDF")
    total_chunks: int = Field(..., description="Total chunks created from document")
    upload_timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 timestamp of upload"
    )


class DocumentChunkMetadata(BaseModel):
    """Metadata associated with an individual text chunk."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    filename: str = Field(..., description="Source filename")
    page: Optional[int] = Field(default=None, description="Source page number (for PDFs)")
    chunk_index: int = Field(..., description="Sequential index of chunk in document")
    text: str = Field(..., description="Raw text content of the chunk")


class DocumentUploadResponse(BaseModel):
    """API response after uploading and indexing a document."""
    document_id: str
    filename: str
    file_type: str
    file_size_bytes: int
    total_chunks: int
    message: str = "Document successfully ingested and indexed."


class DocumentListResponse(BaseModel):
    """List of all indexed documents."""
    total_documents: int
    documents: List[DocumentMetadata]
