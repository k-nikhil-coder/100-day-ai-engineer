"""
Day 28: FastAPI Microservice with Qdrant Vector Search & Metadata Filtering
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "engineering_docs"

app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Connecting to Qdrant server...")
    app_state["client"] = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    print("Loading SentenceTransformer model into memory...")
    app_state["model"] = SentenceTransformer("all-MiniLM-L6-v2")

    print("Day 28 Qdrant search service is ready!")
    yield
    app_state.clear()


app = FastAPI(
    title="Production Vector Search & Metadata Filtering API",
    version="1.0.0",
    lifespan=lifespan,
)


class FilterCriteria(BaseModel):
    category: Optional[str] = Field(
        None, description="Filter by category e.g., backend, databases, algorithms"
    )
    difficulty: Optional[str] = Field(
        None, description="Filter by difficulty e.g., beginner, intermediate, advanced"
    )


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=300)
    top_k: int = Field(3, gt=0, le=5)
    filters: Optional[FilterCriteria] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "how to store and query vectors efficiently",
                    "top_k": 3,
                    "filters": {
                        "category": "databases",
                        "difficulty": None
                    }
                }
            ]
        }
    }


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    client: QdrantClient = app_state.get("client")
    is_connected = False
    try:
        if client:
            client.get_collections()
            is_connected = True
    except Exception:
        is_connected = False

    return {
        "status": "healthy" if is_connected else "degraded",
        "qdrant_connected": is_connected,
    }


@app.post("/search", status_code=status.HTTP_200_OK)
def semantic_search_with_filters(payload: SearchRequest):
    client: QdrantClient = app_state.get("client")
    model: SentenceTransformer = app_state.get("model")

    if not client or not model:
        raise HTTPException(status_code=503, detail="Search dependencies not initialized")

    start_time = time.perf_counter()

    # 1. Embed incoming search query
    query_vector = model.encode(
        [payload.query], convert_to_numpy=True, normalize_embeddings=True
    )[0].tolist()

    # 2. Build Qdrant Metadata Filter conditions dynamically
    must_conditions = []
    if payload.filters:
        if payload.filters.category:
            must_conditions.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=payload.filters.category),
                )
            )
        if payload.filters.difficulty:
            must_conditions.append(
                models.FieldCondition(
                    key="difficulty",
                    match=models.MatchValue(value=payload.filters.difficulty),
                )
            )

    qdrant_filter = models.Filter(must=must_conditions) if must_conditions else None

    # 3. Query Qdrant collection with vector + filter
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=payload.top_k,
    ).points

    matches = []
    for hit in search_results:
        matches.append(
            {
                "id": hit.id,
                "title": hit.payload.get("title"),
                "text": hit.payload.get("text"),
                "category": hit.payload.get("category"),
                "difficulty": hit.payload.get("difficulty"),
                "similarity_score": round(float(hit.score), 4),
            }
        )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

    return {
        "query": payload.query,
        "filters_applied": payload.filters.model_dump() if payload.filters else None,
        "results_count": len(matches),
        "latency_ms": latency_ms,
        "matches": matches,
    }