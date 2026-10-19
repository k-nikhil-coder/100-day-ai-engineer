import os
from fastapi.testclient import TestClient
import main
from main import app, generate_cache_key
import train

if not os.path.exists("./models/model.joblib"):
    train.train_and_export()


class FakeRedis:
    """Lightweight in-memory mock to test Redis hit/miss deterministically."""
    def __init__(self):
        self.store = {}

    def ping(self):
        return True

    def get(self, key: str):
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str):
        self.store[key] = value

    def close(self):
        pass


def test_deterministic_cache_key():
    key1 = generate_cache_key(4.5, 25.0, 5.0)
    key2 = generate_cache_key(4.5, 25.0, 5.0)
    key3 = generate_cache_key(5.0, 25.0, 5.0)
    assert key1 == key2
    assert key1 != key3
    assert key1.startswith("pred:v1:")


def test_cache_miss_then_hit_flow():
    with TestClient(app) as client:
        # Inject FakeRedis to verify caching behavior
        main.redis_client = FakeRedis()

        payload = {
            "property_id": 101,
            "med_inc": 4.5,
            "house_age": 25.0,
            "ave_rooms": 5.0,
        }

        # 1st Request -> Cache Miss (runs model)
        res1 = client.post("/predict", json=payload)
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["cached"] is False

        # 2nd Request (Same features, different property_id) -> Cache Hit!
        payload["property_id"] = 102
        res2 = client.post("/predict", json=payload)
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["cached"] is True
        assert data2["predicted_price_usd"] == data1["predicted_price_usd"]


def test_graceful_fallback_when_redis_down():
    with TestClient(app) as client:
        main.redis_client = None  # Simulate Redis outage

        payload = {
            "property_id": 500,
            "med_inc": 6.2,
            "house_age": 15.0,
            "ave_rooms": 6.5,
        }
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        assert res.json()["cached"] is False