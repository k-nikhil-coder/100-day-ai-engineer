# Day 24: Continuous Integration & Automated Testing for ML Microservices

A production-grade Continuous Integration (CI) pipeline built with GitHub Actions to enforce automated unit testing via `pytest`, strict Pydantic payload validation, and deterministic Docker container builds on every commit and pull request.


---

## 📌 Architecture & Quality Gate Overview

Deploying machine learning models behind web services demands automated software reliability. In a distributed engineering workflow, manually testing endpoints or running ad-hoc local Docker builds risks leaking schema breaking changes, invalid boundary conditions, and broken dependencies into production branches.

This repository enforces an automated CI quality gate where code pushes directly trigger an isolated `ubuntu-latest` runner that acts as an automated gatekeeper:

```text
 Developer Commit / Pull Request (main)
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│  GitHub Actions Runner (ubuntu-latest)                 │
│                                                        │
│  1. actions/checkout@v4                                │
│     └── Fetches repository source tree                 │
│                                                        │
│  2. actions/setup-python@v5                            │
│     └── Provisions Python 3.11 with pip caching        │
│                                                        │
│  3. Install Dependencies                               │
│     └── pip install -r day-24/requirements.txt         │
│                                                        │
│  4. Execute Automated Test Suite (pytest)              │
│     ├── GET  /health           -> 200 OK               │
│     ├── POST /predict (Valid)   -> 200 OK + Calc Check  │
│     └── POST /predict (Invalid) -> 422 Unprocessable   │
│                                                        │
│  5. Containerization Build Gate                        │
│     └── docker build -t ml-valuation-api:latest .      │
└───────────────────────────┬────────────────────────────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
       All Checks Pass            Any Step Fails
     ✅ Merge Allowed           ❌ PR Blocked / Alerts

```

---

## 📂 Project Structure

```text
100-day-ai-engineer/
├── .github/
│   └── workflows/
│       └── ci.yml             # Global CI workflow definition
└── day-24/
    ├── main.py                # FastAPI inference service & schemas
    ├── test_api.py            # Automated test suite (FastAPI TestClient + pytest)
    ├── Dockerfile             # Multi-stage optimized container image
    ├── requirements.txt       # Production & testing dependencies
    ├── pytest.ini             # Pytest warning suppression configuration
    └── README.md              # Engineering and architecture documentation

```

---

## ⚙️ Core Technical Components

### 1. Robust Schema Validation (`main.py`)

Inference payloads are validated strictly at runtime using Pydantic field constraints to guard model inputs against out-of-bounds features:

* `property_id`: Must be strictly positive (`gt=0`).
* `med_inc`: Positive float representing median income (`gt=0`).
* `house_age`: Validated realistic age range between 0 and 120 years (`ge=0, le=120`).
* `ave_rooms`: Must be positive (`gt=0`).

### 2. Comprehensive Test Suite (`test_api.py`)

Utilizes FastAPI's `TestClient` (backed by Starlette) to test application behavior synchronously without binding to a physical network socket:

* **Healthcheck Contract:** Asserts `GET /health` returns HTTP 200 and healthy status payload.
* **Deterministic Inference:** Asserts `POST /predict` produces mathematically expected predictions given clean feature sets.
* **Failure Boundary Rejection:** Submits negative and out-of-bound payloads, asserting the API rejects bad data with `HTTP 422 Unprocessable Entity` rather than throwing internal `500` server errors.

### 3. Container Integrity Verification

The CI runner compiles the `Dockerfile` directly inside the Ubuntu runner. If a dependency introduces an operating system conflict, missing shared C library, or syntax issue, the step fails immediately before runtime deployment.

---

## 🤖 CI Workflow Definition (`.github/workflows/ci.yml`)

The root-level GitHub Actions runner executes the following steps:

```yaml
name: ML Microservice CI

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  test-and-lint:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r day-24/requirements.txt

      - name: Run Pytest Suite
        run: |
          cd day-24
          pytest test_api.py -v

      - name: Build Docker Image
        run: |
          cd day-24
          docker build -t ml-valuation-api:latest .

```

---

## 💻 Local Execution & Verification

### 1. Environment Setup

```bash
cd day-24
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt

```

### 2. Run Test Suite

```bash
pytest test_api.py -v

```

*Expected output:*

```text
collected 3 items

test_api.py::test_health_check PASSED                                    [ 33%]
test_api.py::test_prediction_valid_payload PASSED                         [ 66%]
test_api.py::test_prediction_invalid_payload_fails PASSED                 [100%]

============================== 3 passed in 0.41s ==============================

```

### 3. Build & Run Docker Container Locally

```bash
# Build the image
docker build -t ml-valuation-api:latest .

# Run the container
docker run -d -p 8000:8000 --name valuation_service ml-valuation-api:latest

# Test the health endpoint
curl http://localhost:8000/health

```

---

