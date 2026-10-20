


# Day 27: Semantic Document Search & Dense Retrieval Engine

A production-grade natural language search microservice utilizing `sentence-transformers` (`all-MiniLM-L6-v2`) to generate 384-dimensional dense vector embeddings and **FAISS (Facebook AI Similarity Search)** for sub-millisecond nearest-neighbor cosine similarity retrieval.

---

## 📌 Architecture & Retrieval Flow

Traditional keyword search fails when users search by *meaning* rather than exact strings. This system converts text documents into dense semantic vectors and performs high-speed similarity search in memory.

```text
  [Offline Indexing Pipeline]
        │
        ├── 1. Load Technical Corpus (MLOps, Caching, CI/CD, Algorithms)
        ├── 2. Generate 384-d Embeddings (all-MiniLM-L6-v2 + L2 Normalization)
        └── 3. Build & Serialize FAISS Index (IndexFlatIP for Cosine Similarity)
                     │
                     ▼
  [Online Semantic Search Microservice]
        │
        ├── 1. Container Boot (Docker: python:3.11-slim, CPU-optimized PyTorch)
        ├── 2. FastAPI Lifespan Hook (Loads Model & FAISS Index into Memory once)
        ├── 3. Active Endpoints
        │      ├── GET  /health -> Checks service vitality & indexed doc count
        │      └── POST /search -> Embeds query on the fly, queries FAISS index
        └── 4. Pytest Quality Gate
               └── Validates API health and semantic ranking precision

```

---

## 📂 Project Structure

```text
day-27/
├── models/
│   ├── faiss.index           # Serialized FAISS vector index
│   └── corpus.joblib         # Document metadata mapping corpus
├── main.py                   # FastAPI semantic search API with latency tracking
├── index_docs.py             # Offline vectorization and indexing script
├── test_api.py               # Automated pytest suite verifying semantic ranking
├── requirements.txt          # Production and testing dependencies
├── Dockerfile                # CPU-optimized container build recipe
└── README.md                 # System architecture & documentation

```

---

## ⚙️ Core Technical Highlights

### 1. Inner Product for Cosine Similarity (`IndexFlatIP`)

Documents and search queries are L2-normalized upon embedding. By utilizing FAISS's `IndexFlatIP` (Inner Product), matrix multiplication on normalized vectors yields exact cosine similarity scores in O(N) time with minimal overhead.

### 2. Semantic Ranking Resilience

Tested via `pytest`, the search engine successfully resolves natural language queries (e.g., *"how to speed up slow API responses"*) to conceptual matches like `"Redis Caching Pattern"` even when zero exact keywords overlap between query and title.

### 3. Empirical Latency Telemetry

Every `/search` response returns exact execution telemetry (`latency_ms`), allowing monitoring layers to track retrieval performance under concurrent loads.

---

## 💻 Local Setup & Execution

### 1. Install Dependencies & Build Index

```bash
pip install -r requirements.txt
python index_docs.py

```

### 2. Run Automated Tests

```bash
pytest test_api.py -v

```

### 3. Run the FastAPI Server

```bash
uvicorn main:app --reload

```

Navigate to **`http://localhost:8000/docs`** to test queries interactively.

---

## 🐳 Docker Deployment

### 1. Build Container Image

```bash
docker build -t semantic-search-api:day27 .

```

### 2. Run Container

```bash
docker run -d -p 8000:8000 --name semantic_search_service semantic-search-api:day27

```

### 3. Query the Search API

```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "automated testing and container builds", "top_k": 2}'

```

```json
{
  "query": "automated testing and container builds",
  "results_count": 2,
  "latency_ms": 12.45,
  "matches": [
    {
      "id": 3,
      "title": "GitHub Actions CI/CD",
      "text": "Automated quality gates in CI pipelines run unit tests via pytest and verify docker container builds on every pull request.",
      "similarity_score": 0.8142
    }
  ]
}

```

```

