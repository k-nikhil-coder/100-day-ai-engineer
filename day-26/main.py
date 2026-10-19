import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import redis
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "./models/model.joblib")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 3600))

model_bundle: Dict[str, Any] = {}
redis_client: Optional[redis.Redis] = None


def generate_cache_key(med_inc: float, house_age: float, ave_rooms: float) -> str:
    """Creates a deterministic SHA-256 hash from rounded feature inputs."""
    canonical_features = {
        "med_inc": round(med_inc, 4),
        "house_age": round(house_age, 4),
        "ave_rooms": round(ave_rooms, 4),
    }
    serialized = json.dumps(canonical_features, sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"pred:v1:{digest}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client

    # 1. Load ML Model Artifact
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Missing model artifact at: {MODEL_PATH}. Run train.py first.")

    loaded_data = joblib.load(MODEL_PATH)
    model_bundle["model"] = loaded_data["model"]
    model_bundle["metadata"] = {
        "features": loaded_data["features"],
        "metrics": loaded_data["metrics"],
        "trained_at": loaded_data["trained_at"],
    }

    # 2. Initialize Redis Connection with Graceful Fallback
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=1.0,
        )
        client.ping()
        redis_client = client
        print(f"Connected to Redis cache at {REDIS_HOST}:{REDIS_PORT}")
    except redis.RedisError as exc:
        redis_client = None
        print(f"Redis unavailable ({exc}). Proceeding in uncached fallback mode.")

    yield

    model_bundle.clear()
    if redis_client is not None:
        redis_client.close()


app = FastAPI(
    title="High-Throughput Cached ML Inference API",
    version="2.0.0",
    lifespan=lifespan,
)


class ValuationPayload(BaseModel):
    property_id: int = Field(..., gt=0)
    med_inc: float = Field(..., gt=0)
    house_age: float = Field(..., ge=0, le=120)
    ave_rooms: float = Field(..., gt=0)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    is_model_ready = "model" in model_bundle
    cache_online = False
    if redis_client is not None:
        try:
            cache_online = bool(redis_client.ping())
        except redis.RedisError:
            cache_online = False

    return {
        "status": "healthy" if is_model_ready else "degraded",
        "model_loaded": is_model_ready,
        "redis_connected": cache_online,
    }


@app.post("/predict", status_code=status.HTTP_200_OK)
def predict(payload: ValuationPayload):
    if "model" not in model_bundle:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready to serve predictions",
        )

    start_time = time.perf_counter()
    cache_key = generate_cache_key(
        payload.med_inc, payload.house_age, payload.ave_rooms
    )

    # 1. Check Cache (Cache-Aside Read)
    if redis_client is not None:
        try:
            cached_val = redis_client.get(cache_key)
            if cached_val is not None:
                latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
                return {
                    "property_id": payload.property_id,
                    "predicted_price_usd": float(cached_val),
                    "cached": True,
                    "latency_ms": latency_ms,
                    "status": "success",
                }
        except redis.RedisError:
            pass  # Degrade gracefully on transient Redis read failure

    # 2. Cache Miss: Execute Model Inference
    features_df = pd.DataFrame(
        [
            {
                "MedInc": payload.med_inc,
                "HouseAge": payload.house_age,
                "AveRooms": payload.ave_rooms,
            }
        ]
    )
    features_df = features_df[model_bundle["metadata"]["features"]]

    raw_prediction = model_bundle["model"].predict(features_df)[0]
    scaled_price_usd = round(float(raw_prediction * 100000.0), 2)

    # 3. Populate Cache with TTL
    if redis_client is not None:
        try:
            redis_client.setex(cache_key, CACHE_TTL_SECONDS, str(scaled_price_usd))
        except redis.RedisError:
            pass

    latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
    return {
        "property_id": payload.property_id,
        "predicted_price_usd": scaled_price_usd,
        "cached": False,
        "latency_ms": latency_ms,
        "status": "success",
    }