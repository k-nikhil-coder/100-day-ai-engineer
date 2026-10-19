


# Day 26: Distributed Inference Caching with Redis (Cache-Aside Pattern)

A low-latency machine learning inference microservice implementing deterministic SHA-256 feature hashing and Redis distributed caching (`redis:7-alpine`) to achieve a **21.8x latency reduction** (from `31.3 ms` down to `1.4 ms`) on repeated prediction payloads.


---

## 📌 Architecture & Cache-Aside Flow

In high-throughput production environments, identical feature vectors frequently hit inference endpoints within short time windows. Re-computing DataFrame transformations and model forward passes on duplicate inputs wastes CPU compute and inflates $p95/p99$ tail latency.

This service implements the **Cache-Aside Pattern** with graceful degradation:

```text
               Incoming POST /predict
                         │
                         ▼
       ┌──────────────────────────────────┐
       │ Canonical SHA-256 Feature Hash   │
       │ key = "pred:v1:<sha256_digest>"  │
       └─────────────────┬────────────────┘
                         │
                         ▼
               ┌───────────────────┐
               │ Query Redis Cache │
               └─────────┬─────────┘
                         │
            ┌────────────┴────────────┐
            ▼ (Cache Hit: ~1.4 ms)    ▼ (Cache Miss: ~31.3 ms)
  ┌───────────────────┐     ┌─────────────────────────────┐
  │ Return Prediction │     │ 1. Build Pandas DataFrame   │
  │ "cached": true    │     │ 2. Scikit-Learn Inference   │
  └───────────────────┘     │ 3. SETEX Redis (TTL: 3600s) │
                            │ 4. Return "cached": false   │
                            └─────────────────────────────┘

```

---

## ⚡ Empirical Latency Benchmark

Tested via multi-container Docker Compose deployment (`ml_cached_inference_api` + `ml_redis_cache`):

| Request State | Cache Status | Execution Path | Measured Latency | Speedup |
| --- | --- | --- | --- | --- |
| **1st Request (Cold)** | `"cached": false` | Pydantic $\rightarrow$ Pandas $\rightarrow$ Ridge Model $\rightarrow$ Redis Write | **31.308 ms** | $1.0\times$ (Baseline) |
| **2nd Request (Warm)** | `"cached": true` | Pydantic $\rightarrow$ SHA-256 Hash $\rightarrow$ Redis Memory Read | **1.434 ms** | **$21.8\times$ Faster (95.4% reduction)** |

---

## 📂 Project Structure

```text
day-26/
├── models/
│   └── model.joblib          # Serialized Ridge regression bundle
├── main.py                   # FastAPI service with SHA-256 hashing & Redis cache-aside
├── train.py                  # Offline training and artifact export script
├── test_api.py               # Automated pytest suite with FakeRedis & outage simulation
├── Dockerfile                # Application container build recipe
├── docker-compose.yml        # Multi-container orchestration (FastAPI + Redis 7 Alpine)
├── requirements.txt          # Python dependencies
└── README.md                 # Architecture and benchmark documentation

```

---

## ⚙️ Core Engineering Highlights

### 1. Deterministic Feature Hashing (`generate_cache_key`)

Floating-point features (`med_inc`, `house_age`, `ave_rooms`) are rounded to 4 decimal places, serialized into canonical key-sorted JSON, and hashed via SHA-256 (`pred:v1:<digest>`). Entity identifiers (`property_id`) are intentionally excluded from the hash so identical feature profiles share cached predictions across different request IDs.

### 2. Time-To-Live (TTL) Cache Eviction

Cached inferences are written using Redis `SETEX` with a strict `3600`-second TTL (`CACHE_TTL_SECONDS`), preventing unbounded memory growth and ensuring stale predictions automatically expire after model retraining cycles.

### 3. Fault-Tolerant Graceful Degradation

If the Redis instance becomes unreachable or times out (`socket_connect_timeout=1.0`), the API catches `redis.RedisError` exceptions and seamlessly falls back to live model inference without dropping requests or throwing `HTTP 500` errors.

---

## 💻 Running the Multi-Container Stack

### 1. Launch via Docker Compose

```bash
docker compose up --build -d

```

### 2. Verify Health & Redis Connectivity

```bash
curl http://localhost:8000/health

```

```json
{"status":"healthy","model_loaded":true,"redis_connected":true}

```

### 3. Test Cache Miss vs. Cache Hit

```bash
# 1st Call (Cache Miss -> ~31.3 ms)
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"property_id": 1, "med_inc": 1, "house_age": 120, "ave_rooms": 1}'

# 2nd Call (Cache Hit -> ~1.4 ms)
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"property_id": 1, "med_inc": 1, "house_age": 120, "ave_rooms": 1}'

```

### 4. Run Unit Tests Locally

```bash
pytest test_api.py -v

```

---

