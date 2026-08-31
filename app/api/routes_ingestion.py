from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from app.api.dependencies import get_doc_service_dep
from app.schemas.document import (
    DocumentChunkMetadata,
    DocumentListResponse,
    DocumentMetadata,
    DocumentUploadResponse,
)
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Document Ingestion & Management"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a PDF, TXT, or Markdown document"
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (.pdf, .txt, .md)"),
    chunk_size: Optional[int] = Form(None, description="Optional custom chunk size in tokens/characters"),
    chunk_overlap: Optional[int] = Form(None, description="Optional custom chunk overlap"),
    doc_service: DocumentService = Depends(get_doc_service_dep)
):
    """
    Ingest, parse, chunk, embed, and index an uploaded document into the FAISS vector store.
    """
    return await doc_service.upload_and_process_document(
        file=file,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all indexed documents"
)
async def list_documents(doc_service: DocumentService = Depends(get_doc_service_dep)):
    """Retrieve metadata for all currently indexed documents in the FAISS vector store."""
    docs = doc_service.list_documents()
    return DocumentListResponse(total_documents=len(docs), documents=docs)


@router.get(
    "/{document_id}",
    response_model=DocumentMetadata,
    summary="Get document details by ID"
)
async def get_document_by_id(
    document_id: str,
    doc_service: DocumentService = Depends(get_doc_service_dep)
):
    """Retrieve metadata for a single document specified by its document_id."""
    return doc_service.get_document(document_id)


@router.get(
    "/{document_id}/chunks",
    response_model=List[DocumentChunkMetadata],
    summary="Get all chunks of a document"
)
async def get_document_chunks(
    document_id: str,
    doc_service: DocumentService = Depends(get_doc_service_dep)
):
    """Retrieve all chunk records generated for a specific document."""
    return doc_service.get_document_chunks(document_id)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete a document and purge its vectors"
)
async def delete_document(
    document_id: str,
    doc_service: DocumentService = Depends(get_doc_service_dep)
):
    """Delete a document and rebuild the vector store index without its chunks."""
    deleted = doc_service.delete_document(document_id)
    return {"message": f"Document '{document_id}' successfully deleted.", "deleted": deleted}
