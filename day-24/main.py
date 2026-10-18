"""
Day 24: Testable ML Valuation Service for Automated CI Pipelines
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

app = FastAPI(title="CI Testable Valuation API", version="1.0.0")


class ValuationPayload(BaseModel):
    property_id: int = Field(..., gt=0)
    med_inc: float = Field(..., gt=0)
    house_age: float = Field(..., ge=0, le=120)
    ave_rooms: float = Field(..., gt=0)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy"}


@app.post("/predict", status_code=status.HTTP_200_OK)
def predict_price(payload: ValuationPayload):
    # Deterministic valuation logic for test verification
    price = round((payload.med_inc * 45000.0) + (payload.house_age * 1200.0), 2)
    return {
        "property_id": payload.property_id,
        "predicted_price_usd": price,
        "status": "success"
    }
