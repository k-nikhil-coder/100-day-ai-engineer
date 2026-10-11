import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import numpy as np
import xgboost as xgb

# 1. Application Initialization
app = FastAPI(
    title="California Housing Valuation API",
    description="Production-ready REST API serving an XGBoost model with Pydantic schema validation.",
    version="1.0.0"
)

# 2. Model Loading
MODEL_PATH = "xgb_model.json"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Artifact not found at {MODEL_PATH}. Train or download it first.")

model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)


# 3. Pydantic Schemas
class HousingInput(BaseModel):
    med_inc: float = Field(..., gt=0, description="Median income in block group (in $10,000s)")
    house_age: float = Field(..., ge=0, le=120, description="Median house age in years")
    ave_rooms: float = Field(..., gt=0, description="Average rooms per household")
    ave_bedrms: float = Field(..., gt=0, description="Average bedrooms per household")
    population: float = Field(..., gt=0, description="Block group population")
    ave_occup: float = Field(..., gt=0, description="Average house occupants")
    latitude: float = Field(..., ge=32.0, le=42.5, description="Valid California latitude range")
    longitude: float = Field(..., ge=-125.0, le=-114.0, description="Valid California longitude range")

    class Config:
        json_schema_extra = {
            "example": {
                "med_inc": 8.32,
                "house_age": 41.0,
                "ave_rooms": 6.98,
                "ave_bedrms": 1.02,
                "population": 322.0,
                "ave_occup": 2.55,
                "latitude": 37.88,
                "longitude": -122.23
            }
        }


class PredictionResponse(BaseModel):
    predicted_price_usd: float
    currency: str = "USD"
    status: str = "success"


# 4. Endpoints
@app.get("/", status_code=status.HTTP_200_OK)
def home():
    """Welcome route confirming server health and docs location."""
    return {
        "message": "California Housing API is live. Visit /docs to test endpoints.",
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Service and model readiness probe."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "engine": "XGBoost"
    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_housing_price(payload: HousingInput):
    """Generate home valuation prediction from validated payload."""
    try:
        features = np.array([[
            payload.med_inc,
            payload.house_age,
            payload.ave_rooms,
            payload.ave_bedrms,
            payload.population,
            payload.ave_occup,
            payload.latitude,
            payload.longitude
        ]], dtype=np.float32)

        raw_prediction = model.predict(features)[0]
        estimated_usd = round(float(raw_prediction) * 100_000, 2)

        return PredictionResponse(
            predicted_price_usd=estimated_usd,
            currency="USD",
            status="success"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {str(e)}"
        )
