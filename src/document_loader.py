"""
Document Loader Module.

Extracts text page-by-page from PDF (PyMuPDF) and TXT files, preserving metadata
including source filename and 1-indexed page numbers.
"""

import os
from typing import List, Tuple
import fitz  # PyMuPDF
from langchain_core.documents import Document


def process_file_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Extracts text from document bytes (PDF or TXT) page-by-page and returns
    a list of LangChain Document objects with preserved metadata.

    Metadata schema:
      PDF: {"source": filename, "page": 1-based page number}
      TXT: {"source": filename, "page": None}

    Args:
        file_bytes (bytes): Raw bytes of the uploaded file.
        filename (str): Name of the file.

    Returns:
        List[Document]: Extracted page content documents with metadata.

    Raises:
        ValueError: If file format is unsupported or file is empty/corrupt.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext not in [".pdf", ".txt"]:
        raise ValueError(
            f"Unsupported file type '{ext}'. Please upload PDF or TXT documents."
        )

    if not file_bytes or len(file_bytes.strip()) == 0:
        raise ValueError(f"File '{filename}' is empty.")

    documents = []

    if ext == ".pdf":
        try:
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            total_pages = len(pdf_doc)

            if total_pages == 0:
                raise ValueError(f"PDF file '{filename}' contains 0 pages.")

            extracted_any_text = False
            for page_idx in range(total_pages):
                page = pdf_doc.load_page(page_idx)
                page_text = page.get_text("text").strip()

                if page_text:
                    extracted_any_text = True
                    metadata = {
                        "source": filename,
                        "page": page_idx + 1
                    }
                    documents.append(
                        Document(page_content=page_text, metadata=metadata)
                    )

            pdf_doc.close()

            if not extracted_any_text:
                raise ValueError(
                    f"No extractable text found in PDF '{filename}'. "
                    "It might be a scanned document or image-only PDF."
                )

        except fitz.FileDataError as e:
            raise ValueError(f"Corrupt PDF file '{filename}': {str(e)}") from e
        except ValueError:
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            raise ValueError(f"Error extracting PDF '{filename}': {str(e)}") from e

    elif ext == ".txt":
        try:
            try:
                text_content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text_content = file_bytes.decode("latin-1")

            text_content = text_content.strip()
            if not text_content:
                raise ValueError(f"TXT file '{filename}' is empty.")

            metadata = {
                "source": filename,
                "page": None
            }
            documents.append(
                Document(page_content=text_content, metadata=metadata)
            )

        except ValueError:
            raise
        except Exception as e:  # pylint: disable=broad-exception-caught
            raise ValueError(f"Error reading TXT file '{filename}': {str(e)}") from e

    return documents


def load_documents_from_uploaded_files(
    uploaded_files: List[object]
) -> Tuple[List[Document], List[str]]:
    """
    Processes multiple Streamlit UploadedFile objects.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        Tuple[List[Document], List[str]]:
            - List of successfully processed Document objects.
            - List of error messages for any files that failed processing.
    """
    all_documents = []
    errors = []

    for uploaded_file in uploaded_files:
        try:
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name
            docs = process_file_bytes(file_bytes, filename)
            all_documents.extend(docs)
        except Exception as e:  # pylint: disable=broad-exception-caught
            errors.append(str(e))

    return all_documents, errors
