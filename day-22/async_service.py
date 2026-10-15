"""
Day 22: Asynchronous Inference & Background Task Processing
Decouples user-facing response times from I/O-heavy logging and drift tracking.
"""

import time
import json
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, status
from pydantic import BaseModel, Field

app = FastAPI(
    title="Async Housing Inference Service",
    description="Decouples fast model inference from heavy background logging/monitoring tasks.",
    version="1.0.0"
)

# 1. Schemas
class PredictionPayload(BaseModel):
    property_id: int = Field(..., gt=0)
    med_inc: float = Field(..., gt=0)
    house_age: float = Field(..., ge=0, le=120)
    ave_rooms: float = Field(..., gt=0)

class PredictionResponse(BaseModel):
    property_id: int
    predicted_price_usd: float
    status: str
    processed_at: str


# 2. Simulated Background Worker Task
def log_inference_for_drift_analysis(payload: dict, prediction: float):
    """
    Simulates writing incoming payloads to disk or an audit stream
    to enable Day 21 drift detection tests asynchronously.
    """
    # Simulate slight I/O latency (e.g., writing to DB or message broker)
    time.sleep(0.5)
    
    log_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "payload": payload,
        "prediction": prediction
    }
    
    # Append to a local JSON lines log file
    with open("inference_audit.jsonl", "a") as f:
        f.write(json.dumps(log_record) + "\n")
        
    print(f"[ASYNC WORKER] Logged audit record for property_id={payload['property_id']}")


# 3. Endpoints
@app.post("/predict-async", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_and_audit(payload: PredictionPayload, background_tasks: BackgroundTasks):
    """
    Computes inference immediately and schedules disk I/O / monitoring
    as a non-blocking background task.
    """
    # Fast in-memory calculation (or model.predict)
    mock_prediction = round((payload.med_inc * 45000) + (payload.house_age * 1200), 2)
    
    # Offload the slow/heavy operation to run AFTER the response is returned
    background_tasks.add_task(
        log_inference_for_drift_analysis,
        payload=payload.model_dump(),
        prediction=mock_prediction
    )
    
    # Return response immediately without waiting for background logging
    return PredictionResponse(
        property_id=payload.property_id,
        predicted_price_usd=mock_prediction,
        status="completed",
        processed_at=datetime.utcnow().isoformat()
    )