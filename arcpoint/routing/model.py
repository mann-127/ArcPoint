"""Train Random Forest model to predict future request latency.

This module trains a RandomForestRegressor on synthetic time-series traffic data.
Uses temporal train/test split (no leakage) to predict latency 5 minutes ahead.

Model artifact saved to: models/latency_predictor.pkl
"""

import logging
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_OUTPUT_PATH = "models/latency_predictor.pkl"
DATA_PATH = "data/historical_logs.csv"


def train():
    """Train and evaluate latency prediction model.

    Loads synthetic data, performs temporal train/test split,
    trains RandomForest, and saves to disk.
    """
    logger.info("Loading training data...")
    if not os.path.exists(DATA_PATH):
        logger.error(f"{DATA_PATH} not found. Run 'python data/generate.py' first.")
        return

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} records")

    # Predict latency 5 minutes ahead
    df["target_future_latency"] = df["avg_latency_ms"].shift(-5)
    df["latency_ma_5"] = df["avg_latency_ms"].rolling(window=5).mean()
    df["latency_slope"] = df["avg_latency_ms"].diff()
    df = df.dropna()

    features = ["current_load", "latency_ma_5", "latency_slope"]
    X = df[features]
    y = df["target_future_latency"]

    # Time-based split for time-series data (no leakage)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"Training on {len(X_train)} samples, validating on {len(X_test)}...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    logger.info("Model training complete")
    logger.info(f"MAE: {mae:.2f}ms")
    logger.info(f"R² Score: {r2:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)
    logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    train()
