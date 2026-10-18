# Day 25: Phase 1 Capstone & Model Artifact Serving Pipeline

A production-grade ML inference microservice integrating offline model training, serialized artifact packaging with explicit metadata, modern FastAPI lifespan resource loading, and containerized deployment.


---

## 📌 Architecture & System Flow

Bridging offline model training and online production serving requires strict contracts. Relying on hardcoded weights or raw scripts risks schema mismatch, feature drift, and silent runtime crashes.

This system establishes a decoupled pipeline where an offline regression model is trained, serialized alongside schema metadata, and served via an asynchronous FastAPI service initialized using application lifespans.

```text
  [Offline Training Pipeline]
        │
        ├── 1. Fetch California Housing Dataset (Scikit-Learn / Pandas)
        ├── 2. Fit Ridge Regression Model (Features: MedInc, HouseAge, AveRooms)
        ├── 3. Package Bundle: { model, features, metrics, trained_at (UTC) }
        └── 4. Serialize to disk -> ./models/model.joblib
                     │
                     ▼
  [Online Containerized Microservice]
        │
        ├── 1. Container Boot (Docker: python:3.11-slim)
        ├── 2. FastAPI Lifespan Hook
        │      └── Load ./models/model.joblib into memory once
        ├── 3. Active Endpoints
        │      ├── GET  /health     -> Checks model readiness & process vitality
        │      ├── GET  /model-info -> Exposes evaluation metrics & feature contracts
        │      └── POST /predict    -> Validates schema (Pydantic) & yields predictions
        └── 4. Pytest CI Quality Gate
               └── Automated assertions against health, metadata, and inference

```

---

## 📂 Project Structure

```text
day-25/
├── models/
│   └── model.joblib          # Serialized model artifact + metadata bundle
├── main.py                   # FastAPI application with lifespan management
├── train.py                  # Model training and artifact serialization script
├── test_api.py               # Unit tests verifying lifecycle, metadata, and inference
├── requirements.txt          # Production and testing dependencies
├── Dockerfile                # Container recipe baking model artifacts into image
└── README.md                 # System architecture & deployment guide

```

---

## ⚙️ Core Technical Implementations

### 1. Metadata-Enriched Serialization (`train.py`)

Rather than serializing a raw estimator object, the model is packaged inside a versioned artifact dictionary:

* **Estimator Weights:** Fitted Scikit-Learn `Ridge` regression model.
* **Feature Schema:** Explicit feature ordering `["MedInc", "HouseAge", "AveRooms"]` to eliminate column transposition bugs.
* **Evaluation Baseline:** Training metrics ($R^2$ score and MSE) stored alongside weights.
* **Timestamping:** Strict ISO-8601 timezone-aware UTC timestamps (`datetime.now(timezone.utc)`).

### 2. FastAPI Lifespan Resource Management (`main.py`)

Replaces deprecated `@app.on_event("startup")` hooks with asynchronous context managers:

* Model weights load into memory exactly once at application startup.
* Prevents per-request disk I/O bottlenecks.
* Exposes a dedicated `GET /model-info` endpoint allowing monitoring systems to track what model version is actively serving traffic.

### 3. Schema & Type Alignment

Inference features are cast directly to a named Pandas DataFrame matching training features, eliminating feature-name warnings and runtime type mismatches.

---

## 💻 Local Setup & Execution

### 1. Environment Installation

```bash
pip install -r requirements.txt

```

### 2. Train and Serialize the Artifact

```bash
python train.py

```

*Output:*

```text
Fetching California Housing dataset...
Fitting Ridge regression model...
Model Trained | Test MSE: 0.6589 | R2 Score: 0.4972
Artifact exported successfully to: ./models/model.joblib

```

### 3. Run Automated Tests

```bash
pytest test_api.py -v

```

*Expected output:*

```text
collected 3 items

test_api.py::test_service_lifecycle_and_health PASSED                    [ 33%]
test_api.py::test_model_info_endpoint PASSED                             [ 66%]
test_api.py::test_model_prediction PASSED                                [100%]

============================== 3 passed in 0.41s ==============================

```

---

## 🐳 Docker Deployment

### 1. Build the Container Image

The container build automatically triggers `train.py` to ensure the artifact is compiled directly into the image:

```bash
docker build -t ml-model-service:day25 .

```

### 2. Run the Container

```bash
docker run -d -p 8000:8000 --name valuation_capstone ml-model-service:day25

```

### 3. Query Endpoints

**Health Probe:**

```bash
curl http://localhost:8000/health

```

```json
{"status":"healthy","model_loaded":true}

```

**Model Metadata Contract:**

```bash
curl http://localhost:8000/model-info

```

```json
{
  "features": ["MedInc", "HouseAge", "AveRooms"],
  "metrics": {"test_mse": 0.6589, "r2": 0.4972},
  "trained_at": "2026-10-18T16:30:00+00:00"
}

```

**Inference Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"property_id": 101, "med_inc": 4.5, "house_age": 25.0, "ave_rooms": 5.0}'

```

```json
{
  "property_id": 101,
  "predicted_price_usd": 224530.12,
  "status": "success"
}

```

---
