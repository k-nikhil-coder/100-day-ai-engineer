"""
Day 23: Containerized Async Inference Microservice with Docker Healthchecks
"""

import os
import json
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, status, Response
from pydantic import BaseModel, Field

app = FastAPI(
    title="Containerized Housing Valuation API",
    version="1.0.0"
)

# Persistent log path mounted via Docker volume
LOG_DIR = os.getenv("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)
AUDIT_LOG_FILE = os.path.join(LOG_DIR, "inference_audit.jsonl")


class HousingFeatures(BaseModel):
    property_id: int = Field(..., gt=0)
    med_inc: float = Field(..., gt=0)
    house_age: float = Field(..., ge=0, le=120)
    ave_rooms: float = Field(..., gt=0)


def async_log_payload(payload: dict, prediction: float):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "payload": payload,
        "prediction": prediction
    }
    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check(response: Response):
    """Container healthcheck probe endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/predict", status_code=status.HTTP_200_OK)
def predict(payload: HousingFeatures, background_tasks: BackgroundTasks):
    mock_prediction = round((payload.med_inc * 45000.0) + (payload.house_age * 1150.0), 2)
    
    background_tasks.add_task(
        async_log_payload,
        payload=payload.model_dump(),
        prediction=mock_prediction
    )
    
    return {
        "property_id": payload.property_id,
        "predicted_price_usd": mock_prediction,
        "status": "success"
    }