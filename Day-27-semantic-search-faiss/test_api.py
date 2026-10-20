import os
from fastapi.testclient import TestClient
from main import app
import index_docs

# Ensure vector index exists before running tests
if not os.path.exists("./models/faiss.index"):
    index_docs.build_vector_index()


def test_health_and_index_count():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["indexed_documents"] == 5
        assert data["embedding_dimension"] == 384


def test_semantic_retrieval_relevance():
    with TestClient(app) as client:
        # Query uses zero exact keywords from "Redis Caching Pattern" title
        payload = {
            "query": "how to speed up slow API responses and reduce database load",
            "top_k": 2,
        }
        response = client.post("/search", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["results_count"] == 2
        assert "latency_ms" in data
        # Top match must be Document ID 1 (Redis Caching Pattern)
        top_match = data["matches"][0]
        assert top_match["id"] == 1
        assert top_match["title"] == "Redis Caching Pattern"
        assert top_match["similarity_score"] > 0.4


def test_query_validation_guard():
    with TestClient(app) as client:
        # Query too short (< 3 chars) should trigger 422 Unprocessable Entity
        response = client.post("/search", json={"query": "hi", "top_k": 3})
        assert response.status_code == 422