import pytest
from app.core.exceptions import DocumentValidationError, DocumentProcessingError
from app.ingestion.chunker import DocumentChunker
from app.ingestion.loaders import DocumentLoader
from app.ingestion.parser import TextParser


def test_file_validation():
    """Test filename extension and empty file validation."""
    # Invalid extension
    with pytest.raises(DocumentValidationError):
        DocumentLoader.validate_file("unsupported.exe", b"some binary data")

    # Empty file
    with pytest.raises(DocumentValidationError):
        DocumentLoader.validate_file("empty.txt", b"")

    # Valid extension
    assert DocumentLoader.validate_file("sample.txt", b"Hello world") == ".txt"
    assert DocumentLoader.validate_file("doc.pdf", b"%PDF-1.4...") == ".pdf"
    assert DocumentLoader.validate_file("notes.md", b"# Header") == ".md"


def test_txt_and_md_loading():
    """Test content extraction from TXT and Markdown files."""
    text_content = "Supervised learning is a subfield of artificial intelligence.\nIt uses labeled data."
    doc_id, pages, meta = DocumentLoader.load_document("ai_summary.txt", text_content.encode("utf-8"))

    assert len(doc_id) > 0
    assert meta["file_type"] == ".txt"
    assert len(pages) == 1
    assert "Supervised learning" in pages[0]["text"]


def test_parser_section_extraction():
    """Test text cleaning and Markdown section header parsing."""
    md_content = "# Chapter 1\nIntroduction to RAG systems.\n\n## Section 1.1\nVector databases like FAISS."
    sections = TextParser.extract_sections(md_content)

    assert len(sections) == 2
    assert sections[0]["header"] == "Chapter 1"
    assert "Introduction" in sections[0]["content"]
    assert sections[1]["header"] == "Section 1.1"


def test_chunker_overlap_and_metadata():
    """Test recursive chunking size, overlap, and chunk metadata generation."""
    sample_text = (
        "Artificial intelligence (AI) is intelligence demonstrated by machines. "
        "Machine learning (ML) is the study of computer algorithms that improve automatically through experience. "
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks. "
    ) * 10  # Long string

    page_records = [{"page": 1, "text": sample_text}]
    chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document("doc_123", "test.txt", page_records)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.document_id == "doc_123"
        assert chunk.filename == "test.txt"
        assert chunk.page == 1
        assert len(chunk.text) <= 350
