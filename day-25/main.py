"""
Day 25: Model-Serving Microservice with Lifespan Loading
"""
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "./models/model.joblib")
model_bundle: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once on startup
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Missing model artifact at: {MODEL_PATH}. Run train.py first.")

    loaded_data = joblib.load(MODEL_PATH)
    model_bundle["model"] = loaded_data["model"]
    model_bundle["metadata"] = {
        "features": loaded_data["features"],
        "metrics": loaded_data["metrics"],
        "trained_at": loaded_data["trained_at"],
    }
    print(f"Loaded model bundle successfully from {MODEL_PATH}")

    yield

    # Clean up on shutdown
    model_bundle.clear()


app = FastAPI(
    title="California Housing Model Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


class ValuationPayload(BaseModel):
    property_id: int = Field(..., gt=0)
    med_inc: float = Field(..., gt=0)
    house_age: float = Field(..., ge=0, le=120)
    ave_rooms: float = Field(..., gt=0)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    is_ready = "model" in model_bundle
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
    }


@app.get("/model-info", status_code=status.HTTP_200_OK)
def get_model_info():
    if "metadata" not in model_bundle:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is unavailable",
        )
    return model_bundle["metadata"]


@app.post("/predict", status_code=status.HTTP_200_OK)
def predict(payload: ValuationPayload):
    if "model" not in model_bundle:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready to serve predictions",
        )

    # Frame input with exact columns expected by Scikit-Learn
    features_df = pd.DataFrame(
        [
            {
                "MedInc": payload.med_inc,
                "HouseAge": payload.house_age,
                "AveRooms": payload.ave_rooms,
            }
        ]
    )

    # Enforce training column order strictly
    features_df = features_df[model_bundle["metadata"]["features"]]

    raw_prediction = model_bundle["model"].predict(features_df)[0]
    scaled_price_usd = round(float(raw_prediction * 100000.0), 2)

    return {
        "property_id": payload.property_id,
        "predicted_price_usd": scaled_price_usd,
        "status": "success",
    }
