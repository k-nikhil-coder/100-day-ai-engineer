"""
Automated unit tests executed by GitHub Actions CI pipeline.
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    """Verify health endpoint returns 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prediction_valid_payload():
    """Verify inference returns 200 OK with accurate calculation."""
    payload = {
        "property_id": 101,
        "med_inc": 4.0,
        "house_age": 10.0,
        "ave_rooms": 5.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["property_id"] == 101
    assert data["predicted_price_usd"] == 192000.0
    assert data["status"] == "success"


def test_prediction_invalid_payload_fails():
    """Verify Pydantic schema catches invalid boundary inputs (HTTP 422)."""
    invalid_payload = {
        "property_id": -5,         # Invalid: must be gt 0
        "med_inc": -1.2,          # Invalid: must be gt 0
        "house_age": 250.0,       # Invalid: must be le 120
        "ave_rooms": 0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422