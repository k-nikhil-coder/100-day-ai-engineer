"""
Day 25: Automated unit tests for model serving and metadata endpoints
"""

import os
from fastapi.testclient import TestClient
from main import app
import train

# Guarantee model is trained before running test suite
if not os.path.exists("./models/model.joblib"):
    train.train_and_export()


def test_service_lifecycle_and_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] is True


def test_model_info_endpoint():
    with TestClient(app) as client:
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert data["features"] == ["MedInc", "HouseAge", "AveRooms"]
        assert "test_mse" in data["metrics"]


def test_model_prediction():
    with TestClient(app) as client:
        payload = {
            "property_id": 999,
            "med_inc": 4.5,
            "house_age": 25.0,
            "ave_rooms": 5.0
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["property_id"] == 999
        assert isinstance(data["predicted_price_usd"], float)
        assert data["status"] == "success"
