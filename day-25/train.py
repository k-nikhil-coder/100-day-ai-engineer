"""
Day 25: Baseline Model Training and Artifact Serialization
Trains a Scikit-Learn model on California Housing and exports model.joblib.
"""

import os
import joblib
from datetime import datetime, timezone
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "./models")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")


def train_and_export():
    print("Fetching California Housing dataset...")
    data = fetch_california_housing(as_frame=True)
    
    # Select our target features
    features = ["MedInc", "HouseAge", "AveRooms"]
    X = data.frame[features]
    y = data.target  # Median house value in $100k

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Fitting Ridge regression model...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Model Trained | Test MSE: {mse:.4f} | R2 Score: {r2:.4f}")

    # Package metadata alongside model weights
    artifact = {
        "model": model,
        "features": features,
        "metrics": {"test_mse": round(mse, 4), "r2": round(r2, 4)},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "framework": "scikit-learn"
    }

    joblib.dump(artifact, MODEL_PATH)
    print(f"Artifact exported successfully to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_export()