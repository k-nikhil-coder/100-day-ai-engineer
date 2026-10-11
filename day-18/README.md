# Day 18: FastAPI Model Inference Service

Production REST API wrapper serving an XGBoost California Housing regression model with strict Pydantic input schema validation.

🔗 **Repository Link:** [https://github.com/k-nikhil-coder/100-day-ai-engineer/tree/main/day-18](https://github.com/k-nikhil-coder/100-day-ai-engineer/tree/main/day-18)

---

## 📌 Project Overview

* **Framework:** FastAPI + Uvicorn


* **Validation Layer:** Pydantic v2


* **Model Engine:** XGBoost Regressor


* **Objective:** Move beyond offline notebook evaluation by exposing endpoints with type-checking, range constraints, and automated Swagger documentation.



---

## 📂 Folder Structure

```text
day-18/
├── main.py              # FastAPI application, routes, and validation schemas[cite: 6]
├── xgb_model.json       # Trained XGBoost model artifact
├── requirements.txt     # Python dependencies
└── README.md            # Service documentation

```

---

## ⚙️ Installation & Setup

### 1. Clone & Navigate

```bash
git clone https://github.com/k-nikhil-coder/100-day-ai-engineer.git
cd 100-day-ai-engineer/day-18

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Start the Server

Run Uvicorn with auto-reload:

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000

```

The service will start at `[http://127.0.0.1:8000](http://127.0.0.1:8000)`.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Root status message and documentation links |
| `GET` | `/health` | Health check to verify service and model availability |
| `POST` | `/predict` | Predicts median house valuation based on feature inputs

 |
| `GET` | `/docs` | Interactive Swagger UI for live API testing |

---

## 🧪 Testing the API

### Interactive Docs (Swagger UI)

Open your browser and navigate to:

```text
http://127.0.0.1:8000/docs

```

Use the **Try it out** button on `/predict` to send requests directly from your browser.

### Via cURL (Command Line)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "med_inc": 8.32,
       "house_age": 41.0,
       "ave_rooms": 6.98,
       "ave_bedrms": 1.02,
       "population": 322.0,
       "ave_occup": 2.55,
       "latitude": 37.88,
       "longitude": -122.23
     }'

```

### Expected Response

```json
{
  "predicted_price_usd": 384520.12,
  "currency": "USD",
  "status": "success"
}

```

---

## 🛡️ Input Validation in Action

Pydantic prevents bad inputs from reaching the model:

* **Out-of-range latitude/longitude:** Returns `422 Unprocessable Entity`.
* **Negative age or population:** Rejects execution before running inference.
