import io
from fastapi.testclient import TestClient


def test_health_endpoint(test_client: TestClient):
    """Test /health endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_document_lifecycle_and_query(test_client: TestClient):
    """Test full API workflow: upload, list, get, query, chunks, delete."""
    # 1. Upload document
    sample_text = (
        "Supervised learning is a machine learning paradigm where models are trained on labeled datasets. "
        "Popular algorithms include Decision Trees, Support Vector Machines, and Neural Networks."
    )
    file_bytes = sample_text.encode("utf-8")

    upload_res = test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("supervised_learning.txt", file_bytes, "text/plain")}
    )
    assert upload_res.status_code == 201
    upload_data = upload_res.json()
    doc_id = upload_data["document_id"]
    assert upload_data["filename"] == "supervised_learning.txt"
    assert upload_data["total_chunks"] >= 1

    # 2. List documents
    list_res = test_client.get("/api/v1/documents")
    assert list_res.status_code == 200
    assert list_res.json()["total_documents"] == 1

    # 3. Get document details
    get_res = test_client.get(f"/api/v1/documents/{doc_id}")
    assert get_res.status_code == 200
    assert get_res.json()["document_id"] == doc_id

    # 4. Get chunks
    chunks_res = test_client.get(f"/api/v1/documents/{doc_id}/chunks")
    assert chunks_res.status_code == 200
    assert len(chunks_res.json()) >= 1

    # 5. Query RAG engine
    query_payload = {
        "query": "What is supervised learning?",
        "selected_document_ids": [doc_id],
        "top_k": 5
    }
    query_res = test_client.post("/api/v1/query", json=query_payload)
    assert query_res.status_code == 200
    query_data = query_res.json()
    assert query_data["answer"] is not None
    assert query_data["grounded"] is True

    # 6. Delete document
    del_res = test_client.delete(f"/api/v1/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

    # Confirm deletion
    list_after_del = test_client.get("/api/v1/documents")
    assert list_after_del.json()["total_documents"] == 0
