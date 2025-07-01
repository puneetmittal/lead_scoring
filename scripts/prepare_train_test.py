import pandas as pd
import boto3
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq
import io
import os
import argparse
from botocore.exceptions import ClientError


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--ref-date", type=str, required=True)
    return parser.parse_args()


def get_s3_parquet(bucket: str, key: str):
    s3 = boto3.client("s3")
    buffer = io.BytesIO()
    try:
        s3.download_fileobj(bucket, key, buffer)
        buffer.seek(0)
        return pd.read_parquet(buffer)
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"File not found: s3://{bucket}/{key}")
            return None
        raise


def main():
    args = parse_args()
    ref_date = datetime.strptime(args.ref_date, "%Y-%m-%d")
    train_dates = [(ref_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(6, 3, -1)]
    eval_dates = [(ref_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(3, 1, -1)]
    test_dates = [(ref_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, -1, -1)]

    # Load data
    def load_data(dates):
        dfs = []
        for d in dates:
            key = f"raw/date={d}/lead_data_normalized.parquet"
            df = get_s3_parquet(args.input_path.split('/')[2], key)
            if df is not None:
                dfs.append(df)
        return pd.concat(dfs) if dfs else None

    train_df = load_data(train_dates)
    eval_df = load_data(eval_dates)
    test_df = load_data(test_dates)

    os.makedirs(f"{args.output_path}/train", exist_ok=True)
    os.makedirs(f"{args.output_path}/eval", exist_ok=True)
    os.makedirs(f"{args.output_path}/test", exist_ok=True)

    pq.write_table(pa.Table.from_pandas(train_df), f"{args.output_path}/train/data.parquet")
    pq.write_table(pa.Table.from_pandas(eval_df), f"{args.output_path}/eval/data.parquet")
    pq.write_table(pa.Table.from_pandas(test_df), f"{args.output_path}/test/data.parquet")


if __name__ == "__main__":
    main()