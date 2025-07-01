import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "fsspec", "pyarrow"])

import argparse
import boto3
from datetime import datetime
from botocore.config import Config



def deploy_model(model_s3_uri, endpoint_name, role_arn, region):
    retry_config = Config(
        retries={
            'max_attempts': 1,
            'mode': 'standard'
        }
    )
    sm_client = boto3.client("sagemaker", region_name=region,config = retry_config)
    image_uri = f"683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.3-1"

    # Generate unique names with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"{endpoint_name}-model-{timestamp}"
    config_name = f"{endpoint_name}-config-{timestamp}"

    print(f"Creating model: {model_name}...")
    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_s3_uri,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": model_s3_uri
            }
        },
        ExecutionRoleArn=role_arn
    )

    print(f"Creating endpoint config: {config_name}...")
    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": "ml.m5.large"
            }
        ]
    )

    print(f"Updating endpoint: {endpoint_name}...")
    sm_client.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name
    )

    print("Waiting for endpoint update to complete...")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)

    print(f"Endpoint {endpoint_name} updated successfully!")
    return endpoint_name


def main(model_s3_uri, test_path, threshold, region):
    endpoint_name = "lead-scoring-endpoint-3"
    role_arn = "arn:aws:iam::567510766658:role/SageMakerExecutionRole"


    try:
        deploy_model(model_s3_uri, endpoint_name, role_arn, region)
        print("deployed successfully")
    except Exception as e:
        print(f"Deployment failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-s3-uri", type=str, required=True)
    parser.add_argument("--test-path", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--region", type=str, default="us-east-1")

    args = parser.parse_args()
    main(args.model_s3_uri, args.test_path, args.threshold, args.region)
