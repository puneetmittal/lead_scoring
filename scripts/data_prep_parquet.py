import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import boto3
import pyarrow as pa
import pyarrow.parquet as pq
import sys
import random
import os

random_seed = random.randint(45, 100000)
np.random.seed(random_seed)
BUCKET_NAME = "salesforce-leads-data-puneet"
RAW_FOLDER = "raw"
s3 = boto3.client("s3")

def generate_numeric_data(num_rows: int, day: datetime) -> pd.DataFrame:
    data = {
        "age": np.random.randint(18, 70, size=num_rows),
        "income": np.random.randint(20000, 200000, size=num_rows),
        "pages_visited": np.random.randint(1, 20, size=num_rows),
        "time_on_site": np.random.uniform(10, 500, size=num_rows),
        "conversion_rate": np.random.rand(num_rows),
        "score": np.random.randint(1, 6, size=num_rows),
    }
    for i in range(6, 51):
        data[f"feature_{i}"] = np.random.randn(num_rows)
    return pd.DataFrame(data)

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    y = df["score"]
    X = df.drop(columns=["score"])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled["score"] = y.reset_index(drop=True)
    return X_scaled

def upload_to_local(df: pd.DataFrame, day: datetime, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    date_str = day.strftime("%Y-%m-%d")
    file_path = os.path.join(output_dir, f"lead_data_normalized_{date_str}.parquet")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)
    print(f"Saved file locally to {file_path}")

def main(reference_date: str, output_dir: str):
    try:
        ref_date = datetime.strptime(reference_date, "%Y-%m-%d")
    except ValueError:
        print("Invalid date format. Use YYYY-MM-DD.")
        sys.exit(1)

    for i in range(1, 8):
        day = ref_date - timedelta(days=i)
        df = generate_numeric_data(num_rows=10000, day=day)
        df_normalized = normalize_features(df)
        upload_to_local(df_normalized, day,output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python data_prep_parquet.py <YYYY-MM-DD> <OUTPUT_DIR>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])