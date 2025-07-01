# README: Automated Retraining Pipeline

## Overview

This repository contains an automated retraining pipeline for an XGBoost-based lead scoring model on AWS SageMaker. The pipeline:

* Ingests weekly feature data from S3 (Parquet format)
* Performs preprocessing and splits data
* Trains and evaluates an XGBRegressor model
* Registers and deploys the model to a SageMaker endpoint

---

## Prerequisites

* **AWS account** with SageMaker permissions
* **Python 3.9+**
* **AWS CLI** configured

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/lead-scoring-retraining.git
   cd lead_scoring_pipeline
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv .lead_scoring
   source .lead_scoring/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure AWS credentials**

   ```bash
   aws configure
   ```

---

## Running the Pipeline

### 1. Ad-hoc Run

To trigger the retraining pipeline manually:

```bash
python pipelines/pipeline.py
```

This script will:

* Fetch the last 7 days of Parquet data
* Split data by time-based 4/1/2 day windows
* Preprocess and split the data
* Train the XGBRegressor model
* Evaluate metrics (Spearman)
* Register and deploy the new model to the SageMaker endpoint

### 2. Scheduled Run (Weekly)

You can schedule the pipeline via AWS EventBridge 
The pipeline is already scheduled weekly from AWS EventBridge

---

## Implemented Features

* **Data Format**: Parquet ingestion
* **Time-based Splits**: Train/validate/test on 4/1/2-day windows
* **Model Training**: XGBRegressor 
* **Validation**: Spearman metric
* **Version Control**: SageMaker Model Registry integration
* **Deployment**: Direct endpoint update for new model
* **Monitoring**: CloudWatch metrics for endpoint status

---

## Future Scope

* **Shadow Deployment**: Mirror live traffic to a shadow endpoint for real-world validation and do sanity check before deployment

* **Business Metrics** Tracking: Tracking Business metrics like lead conversion score to measure the impact of model

* **Rollback model**: Rollback to previous version if model not performing to the benchmark

* **Drift Detection**: Automate retraining trigger based on data drift alerts.