import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF

from app.core.config import settings
from app.core.exceptions import DocumentValidationError, DocumentProcessingError
from app.core.logging import logger
from app.ingestion.parser import TextParser


class DocumentLoader:
    """Handles loading, validation, and content extraction for PDF, TXT, and Markdown files."""

    @staticmethod
    def validate_file(filename: str, file_bytes: bytes) -> str:
        """Validate filename extension and file size limit."""
        file_ext = Path(filename).suffix.lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise DocumentValidationError(
                f"Unsupported file format '{file_ext}'. Allowed formats: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )

        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise DocumentValidationError(
                f"File size ({len(file_bytes) / (1024*1024):.2f} MB) exceeds maximum limit of {settings.MAX_FILE_SIZE_MB} MB."
            )

        if len(file_bytes) == 0:
            raise DocumentValidationError("Uploaded file is empty.")

        return file_ext

    @classmethod
    def load_document(cls, filename: str, file_bytes: bytes) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Extract text per page/section from file bytes.
        Returns (document_id, list of page dicts, metadata dict).
        Each page dict: {"page": int or None, "text": str}.
        """
        file_ext = cls.validate_file(filename, file_bytes)
        document_id = str(uuid.uuid4())

        page_records: List[Dict[str, Any]] = []
        total_pages: int | None = None

        try:
            if file_ext == ".pdf":
                page_records, total_pages = cls._extract_pdf(file_bytes)
            elif file_ext in [".txt", ".md"]:
                page_records = cls._extract_text_or_md(file_bytes)
                total_pages = 1
            else:
                raise DocumentValidationError(f"Unsupported extension: {file_ext}")
        except Exception as e:
            logger.error(f"Error processing file '{filename}': {str(e)}", exc_info=True)
            if isinstance(e, DocumentValidationError):
                raise e
            raise DocumentProcessingError(f"Failed to process document '{filename}': {str(e)}")

        if not page_records or all(not p["text"].strip() for p in page_records):
            raise DocumentProcessingError(f"Document '{filename}' contains no readable text.")

        metadata = {
            "document_id": document_id,
            "filename": filename,
            "file_type": file_ext,
            "file_size_bytes": len(file_bytes),
            "total_pages": total_pages,
        }

        return document_id, page_records, metadata

    @staticmethod
    def _extract_pdf(file_bytes: bytes) -> Tuple[List[Dict[str, Any]], int]:
        """Extract text page by page from PDF using PyMuPDF."""
        page_records = []
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)

        for page_idx in range(total_pages):
            page = doc.load_page(page_idx)
            raw_text = page.get_text("text")
            cleaned_text = TextParser.clean_text(raw_text)
            if cleaned_text:
                page_records.append({
                    "page": page_idx + 1,  # 1-indexed for user readability
                    "text": cleaned_text
                })

        doc.close()
        return page_records, total_pages

    @staticmethod
    def _extract_text_or_md(file_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract text from TXT or Markdown files."""
        try:
            raw_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                raw_text = file_bytes.decode("latin-1")
            except Exception as e:
                raise DocumentProcessingError(f"Failed to decode text file encoding: {str(e)}")

        cleaned_text = TextParser.clean_text(raw_text)
        return [{"page": 1, "text": cleaned_text}]
