
# Day 28: Production Vector Database & Metadata Filtering API

A production-grade semantic search microservice leveraging **Qdrant** as an external vector database, combined with `sentence-transformers` (`all-MiniLM-L6-v2`) for dense embeddings, strict payload metadata filtering, and containerized multi-service orchestration via Docker Compose.


---

## Architecture & System Design

Unlike in-memory local indices (like FAISS), production systems require dedicated vector databases that support concurrent clients, persistent storage, and strict payload filtering to combine exact keyword/categorical conditions with dense semantic similarity scores.

```text
  [Offline Indexing & Setup]
        │
        ├── 1. Load Technical Documents & Metadata (Category, Difficulty)
        ├── 2. Generate 384-d Embeddings (all-MiniLM-L6-v2 + L2 Normalization)
        └── 3. Upsert Vectors + Payloads into Qdrant Container (Port 6333)
                     │
                     ▼
  [Online Filtered Vector Search Microservice]
        │
        ├── 1. Multi-Container Docker Stack (Qdrant DB + FastAPI Gateway)
        ├── 2. FastAPI Lifespan Hook (Initializes Qdrant Client & Model)
        ├── 3. Active Endpoints
        │      ├── GET  /health -> Checks Qdrant server vitality & connection
        │      └── POST /search -> Embeds query + applies dynamic Qdrant filters
        └── 4. Pre-configured Swagger UI (Custom JSON Request Schema Pre-filled)

```

---

## 📂 Project Structure

```text
Day-28-qdrant-vector-db-filters/
├── main.py                   # FastAPI service with dynamic Qdrant metadata filtering
├── setup_db.py               # Database initialization and document upsert script
├── requirements.txt          # Pinned production and client dependencies
├── Dockerfile                # CPU-optimized container build recipe for the API
├── docker-compose.yml        # Multi-container orchestration (Qdrant + FastAPI)
└── README.md                 # System architecture & documentation

```

---

## ⚙️ Core Technical Highlights

### 1. Dedicated Vector DB Architecture (`Qdrant`)

Runs a decoupled vector database instance via Docker (`qdrant/qdrant:latest`) with persistent volume storage (`qdrant_storage`), mirroring true microservice production environments.

### 2. Dynamic Payload Metadata Filtering

The FastAPI `/search` endpoint accepts optional filter criteria (`category`, `difficulty`) and translates them on the fly into native Qdrant `models.FieldCondition` rules, ensuring search results only return documents matching strict categorical constraints.

### 3. Pre-configured Swagger UX (`json_schema_extra`)

Integrated custom Pydantic examples so the interactive Swagger docs (`/docs`) automatically pre-fill with valid request payloads, eliminating manual string-matching errors during testing.

---

## 💻 Local Setup & Execution

### 1. Spin up the Multi-Container Stack (Qdrant + FastAPI)

```bash
docker compose up --build -d

```

### 2. Initialize and Populate the Database Collections

```bash
pip install -r requirements.txt
python setup_db.py

```

### 3. Run the FastAPI Server Locally (Alternative to Docker API container)

```bash
uvicorn main:app --reload

```

Navigate to **`http://localhost:8000/docs`** to test filtered semantic searches interactively.

---

## 🔌 API Usage Example

### Request (`POST /search`)

```json
{
  "query": "how to store and query vectors efficiently",
  "top_k": 3,
  "filters": {
    "category": "databases",
    "difficulty": "advanced"
  }
}

```

### Response

```json
{
  "query": "how to store and query vectors efficiently",
  "filters_applied": {
    "category": "databases",
    "difficulty": "advanced"
  },
  "results_count": 1,
  "latency_ms": 34.12,
  "matches": [
    {
      "id": 3,
      "title": "Qdrant Metadata Filtering",
      "text": "Qdrant enables rapid filtered vector search by combining dense similarity scores with strict payload metadata matching conditions.",
      "category": "databases",
      "difficulty": "advanced",
      "similarity_score": 0.4049
    }
  ]
}

```



