import joblib
import os
import json
import pandas as pd
import numpy as np

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "xgb_salesforce_lead_model.joblib"))
    with open(os.path.join(model_dir, "feature_names.json"), "r") as f:
        model.feature_names = json.load(f)
    return model

def input_fn(request_body, content_type):
    if content_type == "application/x-parquet":
        import io
        df = pd.read_parquet(io.BytesIO(request_body))
        return df
    elif content_type == "text/csv":
        import io
        df = pd.read_csv(io.StringIO(request_body))
        return df
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    input_df = input_data[model.feature_names]
    preds = model.predict(input_df)
    return np.round(np.clip(preds, 1, 5)).astype(int)

def output_fn(prediction, content_type="application/json"):
    import json
    return json.dumps({"scores": prediction.tolist()})