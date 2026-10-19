import os
from datetime import datetime, timezone
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "./models")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")


def train_and_export():
    print("Fetching California Housing dataset...")
    data = fetch_california_housing(as_frame=True)

    features = ["MedInc", "HouseAge", "AveRooms"]
    X = data.frame[features]
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    artifact = {
        "model": model,
        "features": features,
        "metrics": {"test_mse": round(mse, 4), "r2": round(r2, 4)},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "framework": "scikit-learn",
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"Artifact exported to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_export()