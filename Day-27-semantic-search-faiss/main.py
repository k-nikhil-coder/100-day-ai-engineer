import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import faiss
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_DIR = os.getenv("ARTIFACT_DIR", "./models")
INDEX_PATH = os.path.join(MODEL_DIR, "faiss.index")
DOCS_PATH = os.path.join(MODEL_DIR, "corpus.joblib")

app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
        raise RuntimeError(
            "FAISS index or corpus missing. Run `python index_docs.py` first."
        )

    print("Loading SentenceTransformer (all-MiniLM-L6-v2) into memory...")
    app_state["model"] = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading FAISS vector index and document corpus...")
    app_state["index"] = faiss.read_index(INDEX_PATH)
    app_state["corpus"] = joblib.load(DOCS_PATH)

    print("Semantic search engine online!")
    yield
    app_state.clear()


app = FastAPI(
    title="Semantic Document Search & Dense Retrieval API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/docs")


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=3, max_length=300)
    top_k: int = Field(3, gt=0, le=5)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    is_ready = "index" in app_state and "model" in app_state
    return {
        "status": "healthy" if is_ready else "degraded",
        "indexed_documents": len(app_state.get("corpus", [])),
        "embedding_dimension": 384 if is_ready else 0,
    }


@app.post("/search", status_code=status.HTTP_200_OK)
def semantic_search(payload: SearchQuery):
    if "index" not in app_state or "model" not in app_state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search engine is not initialized",
        )

    start_time = time.perf_counter()

    model: SentenceTransformer = app_state["model"]
    index: faiss.Index = app_state["index"]
    corpus: List[Dict[str, Any]] = app_state["corpus"]

    # 1. Encode query into normalized 384-d vector
    query_vector = model.encode(
        [payload.query], convert_to_numpy=True, normalize_embeddings=True
    )

    # 2. Nearest-neighbor search over FAISS index
    k = min(payload.top_k, len(corpus))
    scores, indices = index.search(query_vector, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        doc = corpus[idx]
        results.append(
            {
                "id": doc["id"],
                "title": doc["title"],
                "text": doc["text"],
                "similarity_score": round(float(score), 4),
            }
        )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 3)

    return {
        "query": payload.query,
        "results_count": len(results),
        "latency_ms": latency_ms,
        "matches": results,
    }