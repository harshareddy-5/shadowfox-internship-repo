import numpy as np
import pytest
from app.retrieval.faiss_store import FAISSVectorStore
from app.schemas.document import DocumentChunkMetadata, DocumentMetadata


def test_faiss_add_search_filter_delete(temp_data_dirs):
    """Test FAISS vector store lifecycle: add vectors, search, document-scoped filter, delete."""
    uploads_dir, indexes_dir = temp_data_dirs
    store = FAISSVectorStore(dimension=384, index_dir=indexes_dir)

    # Document 1 chunks
    doc1_meta = DocumentMetadata(
        document_id="doc_1",
        filename="ai_paper.pdf",
        file_type=".pdf",
        file_size_bytes=1024,
        total_pages=2,
        total_chunks=2
    )
    chunks_doc1 = [
        DocumentChunkMetadata(chunk_id="c1", document_id="doc_1", filename="ai_paper.pdf", page=1, chunk_index=0, text="Deep learning architectures."),
        DocumentChunkMetadata(chunk_id="c2", document_id="doc_1", filename="ai_paper.pdf", page=2, chunk_index=1, text="Transformer models for NLP.")
    ]
    vecs_doc1 = np.random.randn(2, 384).astype(np.float32)
    # L2 normalize
    vecs_doc1 /= np.linalg.norm(vecs_doc1, axis=1, keepdims=True)

    store.add_chunks(chunks_doc1, vecs_doc1, doc1_meta)
    assert store.index.ntotal == 2

    # Document 2 chunks
    doc2_meta = DocumentMetadata(
        document_id="doc_2",
        filename="python_guide.txt",
        file_type=".txt",
        file_size_bytes=500,
        total_pages=1,
        total_chunks=1
    )
    chunks_doc2 = [
        DocumentChunkMetadata(chunk_id="c3", document_id="doc_2", filename="python_guide.txt", page=1, chunk_index=0, text="Python async programming.")
    ]
    vecs_doc2 = np.random.randn(1, 384).astype(np.float32)
    vecs_doc2 /= np.linalg.norm(vecs_doc2, axis=1, keepdims=True)

    store.add_chunks(chunks_doc2, vecs_doc2, doc2_meta)
    assert store.index.ntotal == 3

    # Document-Scoped Search (Filter for doc_2 only)
    query_vec = vecs_doc2[0:1]
    filtered_results = store.similarity_search(query_vec, top_k=10, selected_document_ids=["doc_2"])
    assert len(filtered_results) == 1
    assert filtered_results[0][0].document_id == "doc_2"

    # Search all documents
    all_results = store.similarity_search(query_vec, top_k=10)
    assert len(all_results) == 3

    # Persistent reload test
    reloaded_store = FAISSVectorStore(dimension=384, index_dir=indexes_dir)
    assert reloaded_store.index.ntotal == 3
    assert len(reloaded_store.get_indexed_documents()) == 2

    # Delete Document 1
    deleted = reloaded_store.delete_document("doc_1")
    assert deleted is True
    assert reloaded_store.index.ntotal == 1
    assert len(reloaded_store.get_indexed_documents()) == 1
    assert reloaded_store.get_indexed_documents()[0].document_id == "doc_2"
