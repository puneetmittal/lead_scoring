import pandas as pd
import joblib
from scipy.stats import spearmanr
import json
import os
import argparse
import tarfile

EVAL_OUTPUT_DIR = "/opt/ml/processing/evaluation"

def extract_model_tar(model_path):
    for file in os.listdir(model_path):
        if file.endswith(".tar.gz"):
            tar_path = os.path.join(model_path, file)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=model_path)
            return
    raise FileNotFoundError("Model archive (.tar.gz) not found in model_path")


def evaluate(test_path, model_path):
    extract_model_tar(model_path)
    # Load data and model
    test_df = pd.read_parquet(os.path.join(test_path, "data.parquet"))
    model = joblib.load(os.path.join(model_path, "xgb_salesforce_lead_model.joblib"))

    # Predict and evaluate
    X_test = test_df.drop("score", axis=1)
    y_true = test_df["score"]
    preds = model.predict(X_test)

    # Calculate metrics
    spearman = spearmanr(y_true, preds)[0]

    # Save results
    metrics = {
        "spearman_correlation": float(spearman),
        "test_samples": len(y_true)
    }
    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(EVAL_OUTPUT_DIR, "evaluation.json"), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.test_path, args.model_path)