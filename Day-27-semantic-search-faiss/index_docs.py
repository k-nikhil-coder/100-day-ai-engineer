import os
import faiss
import joblib
from sentence_transformers import SentenceTransformer

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "./models")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
INDEX_PATH = os.path.join(ARTIFACT_DIR, "faiss.index")
DOCS_PATH = os.path.join(ARTIFACT_DIR, "corpus.joblib")

# Technical engineering corpus across MLOps, Caching, CI/CD, and Algorithms
CORPUS = [
    {
        "id": 1,
        "title": "Redis Caching Pattern",
        "text": "Using Redis with the cache-aside pattern minimizes database load and reduces inference latency from milliseconds to microseconds.",
    },
    {
        "id": 2,
        "title": "FastAPI Lifespan Events",
        "text": "Modern FastAPI applications manage model lifecycle state using async context manager lifespan hooks instead of legacy startup events.",
    },
    {
        "id": 3,
        "title": "GitHub Actions CI/CD",
        "text": "Automated quality gates in CI pipelines run unit tests via pytest and verify docker container builds on every pull request.",
    },
    {
        "id": 4,
        "title": "Pydantic Validation",
        "text": "Pydantic enforces strict runtime type safety and field boundary validation for incoming API payloads and configuration schemas.",
    },
    {
        "id": 5,
        "title": "Binary Search Optimization",
        "text": "Binary search on answer space allows efficient problem solving over monotonic functions in O(log n) time complexity.",
    },
]


def build_vector_index():
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [doc["text"] for doc in CORPUS]
    print(f"Embedding {len(texts)} documents into 384-d vectors...")
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    dimension = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP (dim={dimension})...")

    # Inner Product on L2-normalized vectors is mathematically identical to Cosine Similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    joblib.dump(CORPUS, DOCS_PATH)
    print(f"Success! FAISS index saved to {INDEX_PATH} | Corpus saved to {DOCS_PATH}")


if __name__ == "__main__":
    build_vector_index() 