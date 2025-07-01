import os
import pandas as pd
import xgboost as xgb
import joblib
import json
from datetime import datetime


def load_data(input_path):
    train_files = [f for f in os.listdir(input_path) if f.endswith('.parquet')]
    dfs = [pd.read_parquet(os.path.join(input_path, f)) for f in train_files]
    return pd.concat(dfs, ignore_index=True)


def train():
    input_path = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    df = load_data(input_path)
    X = df.drop("score", axis=1)
    y = df["score"]

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6
    )
    model.fit(X, y)

    # Save artifacts
    joblib.dump(model, os.path.join(model_dir, "xgb_salesforce_lead_model.joblib"))
    with open(os.path.join(model_dir, "feature_names.json"), "w") as f:
        json.dump(list(X.columns), f)

    # Save metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "features": list(X.columns),
        "hyperparameters": model.get_params()
    }
    with open(os.path.join(model_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    train()