"""
Day 28: Qdrant Vector Database Initialization and Payload Upsert
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# Connect to Qdrant instance (local container or localhost)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "engineering_docs"

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

DOCUMENTS = [
    {
        "id": 1,
        "title": "Redis Caching Pattern",
        "text": "Using Redis with the cache-aside pattern minimizes database load and reduces inference latency from milliseconds to microseconds.",
        "category": "backend",
        "difficulty": "intermediate",
    },
    {
        "id": 2,
        "title": "FastAPI Lifespan Events",
        "text": "Modern FastAPI applications manage model lifecycle state using async context manager lifespan hooks instead of legacy startup events.",
        "category": "backend",
        "difficulty": "beginner",
    },
    {
        "id": 3,
        "title": "Qdrant Metadata Filtering",
        "text": "Qdrant enables rapid filtered vector search by combining dense similarity scores with strict payload metadata matching conditions.",
        "category": "databases",
        "difficulty": "advanced",
    },
    {
        "id": 4,
        "title": "Pydantic Validation",
        "text": "Pydantic enforces strict runtime type safety and field boundary validation for incoming API payloads and configuration schemas.",
        "category": "backend",
        "difficulty": "beginner",
    },
    {
        "id": 5,
        "title": "Binary Search Optimization",
        "text": "Binary search on answer space allows efficient problem solving over monotonic functions in O(log n) time complexity.",
        "category": "algorithms",
        "difficulty": "intermediate",
    },
]


def init_vector_database():
    print("Initializing Qdrant client...")

    # Recreate collection cleanly
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=384,  # Dimension for all-MiniLM-L6-v2
            distance=models.Distance.COSINE,
        ),
    )
    print(f"Collection '{COLLECTION_NAME}' created successfully.")

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [doc["text"] for doc in DOCUMENTS]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    print("Upserting vectors and metadata payloads into Qdrant...")
    points = []
    for idx, doc in enumerate(DOCUMENTS):
        points.append(
            models.PointStruct(
                id=doc["id"],
                vector=embeddings[idx].tolist(),
                payload={
                    "title": doc["title"],
                    "text": doc["text"],
                    "category": doc["category"],
                    "difficulty": doc["difficulty"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Successfully indexed {len(points)} documents into Qdrant!")


if __name__ == "__main__":
    init_vector_database()