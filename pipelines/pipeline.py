import boto3
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.model import Model
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.properties import PropertyFile
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

# Configuration
region = boto3.Session().region_name
role = "arn:aws:iam::567510766658:role/SageMakerExecutionRole"
bucket = "salesforce-leads-data-puneet"
pipeline_name = "LeadScoringPipeline-v2"
pipeline_session = PipelineSession()

# 1. Data Generation
data_processor = ScriptProcessor(
    command=["python3"],
    image_uri=f"683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.3-1",
    instance_type="ml.t3.medium",
    instance_count=1,
    role=role,
    base_job_name="data-generation"
)

data_step = ProcessingStep(
    name="GenerateData",
    processor=data_processor,
    outputs=[ProcessingOutput(output_name="raw", source="/opt/ml/processing/output")],
    code="scripts/data_prep_parquet.py",
    job_arguments=["2025-06-19", "/opt/ml/processing/output"]
)

# 2. Train/Eval/Test Split
split_processor = ScriptProcessor(
    command=["python3"],
    image_uri=f"683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.3-1",
    instance_type="ml.t3.medium",
    instance_count=1,
    role=role
)

split_step = ProcessingStep(
    name="SplitData",
    processor=split_processor,
    inputs=[
        ProcessingInput(
            source=data_step.properties.ProcessingOutputConfig.Outputs["raw"].S3Output.S3Uri,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
        ProcessingOutput(output_name="eval", source="/opt/ml/processing/output/eval"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/output/test")
    ],
    code="scripts/prepare_train_test.py",
    job_arguments=[
        "--input-path", f"s3://{bucket}",
        "--output-path", "/opt/ml/processing/output",
        "--ref-date", "2025-06-19"
    ]
)

# 3. Model Training
xgb_estimator = XGBoost(
    entry_point="train.py",
    source_dir="scripts",
    framework_version="1.3-1",
    py_version="py3",
    instance_type="ml.c4.2xlarge",
    instance_count=1,
    role=role,
    output_path=f"s3://{bucket}/models/"
)

train_step = TrainingStep(
    name="TrainModel",
    estimator=xgb_estimator,
    inputs={
        "train": TrainingInput(
            split_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="application/x-parquet"
        )
    }
)

# 4. Model Evaluation
eval_processor = ScriptProcessor(
    command=["python3"],
    image_uri=f"683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.3-1",
    instance_type="ml.t3.medium",
    instance_count=1,
    role=role
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

eval_step = ProcessingStep(
    name="EvaluateModel",
    processor=eval_processor,
    inputs=[
        ProcessingInput(
            source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=split_step.properties.ProcessingOutputConfig.Outputs["eval"].S3Output.S3Uri,
            destination="/opt/ml/processing/eval"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
    ],
    code="scripts/evaluate.py",
    property_files=[evaluation_report],
    job_arguments=[
        "--test-path", "/opt/ml/processing/eval",
        "--model-path", "/opt/ml/processing/model"
    ]
)

# 5. Model Registration
model = Model(
    image_uri=xgb_estimator.training_image_uri(),
    model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    entry_point="inference.py",
    sagemaker_session=pipeline_session,
    source_dir="s3://salesforce-leads-data-puneet/code/scripts.tar.gz"
)

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=eval_step.properties.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri,
        content_type="application/json"
    )
)

register_step = ModelStep(
    name="RegisterModel",
    step_args=model.register(
        model_package_group_name="LeadScoring",
        approval_status="Approved",
        model_metrics=model_metrics,
        content_types=["text/csv", "application/x-parquet"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"]
    ),
    depends_on=[eval_step]
)

# 6. Deploy and Compare Step
deploy_processor = ScriptProcessor(
    command=["python3"],
    image_uri=f"683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.3-1",
    instance_type="ml.t3.medium",
    instance_count=1,
    role=role,
    env={
        "AWS_REGION": region  # Pass the region to the processing job
    }
)

deploy_step = ProcessingStep(
    name="DeployAndCompare",
    processor=deploy_processor,
    inputs=[
        # ProcessingInput(
        #     source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        #     destination="/opt/ml/processing/model"
        # ),
        ProcessingInput(
            source=split_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="comparison", source="/opt/ml/processing/comparison")
    ],
    code="scripts/deploy.py",
    job_arguments=[
        "--model-s3-uri", train_step.properties.ModelArtifacts.S3ModelArtifacts,
        "--test-path", "/opt/ml/processing/test",
        "--threshold", "0.002"
    ]
)

# Create condition for evaluation threshold
cond_gte = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=eval_step.name,
        property_file=evaluation_report,
        json_path="spearman_correlation"
    ),
    right=0.002
)

# Create condition step
condition_step = ConditionStep(
    name="CheckEvaluationThreshold",
    conditions=[cond_gte],
    if_steps=[deploy_step],  # Only deploy if evaluation passes
    # if_steps=[register_step],
    else_steps=[]  # Skip deployment if evaluation fails
)

# Create Pipeline
pipeline = Pipeline(
    name=pipeline_name,
    steps=[data_step, split_step, train_step, eval_step, register_step, condition_step],
    sagemaker_session=pipeline_session
)

# Deploy
pipeline.upsert(role_arn=role)
execution = pipeline.start()
print(f"Pipeline ARN: {execution.arn}")