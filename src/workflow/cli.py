# cli.py
import os
import argparse
import json
import random
import string
from kfp import dsl
from kfp import compiler
import google.cloud.aiplatform as aip


GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
BUCKET_URI = f"gs://{GCS_BUCKET_NAME}"
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/root"
GOOGLE_APP_CREDENTIALS = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
GCP_SERVICE_ACCOUNT = '32226619505-compute@developer.gserviceaccount.com'

DATA_PREPROCESSING_IMAGE = "nevilgeorge/eng-ai-preprocessing"
TENSORIZING_IMAGE = "abzp/eng-ai-tensorizing:abpujare"
AGE_MODEL_TRAINING_IMAGE = "abzp/ac215-age-model-training:abpujare"
MODEL_DEPLOYMENT_IMAGE = "annielcook/engai-model-deployment:1115v1"


def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def main(args=None):
    print("CLI Arguments:", args)
    with open(GOOGLE_APP_CREDENTIALS, 'r') as f:
        secrets = json.load(f)
    
    gcp_project_id = secrets['project_id']

    if args.data_preprocessing:
        # Define a Pipeline
        @dsl.pipeline
        def data_preprocessing_pipeline():
            data_preprocessing_component()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            data_preprocessing_pipeline, package_path="data_preprocessing.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=gcp_project_id, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "engai-preprocessing-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data_preprocessing.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCP_SERVICE_ACCOUNT)

    if args.tensorizing:
        # Define a Pipeline
        @dsl.pipeline
        def tensorizing_pipeline():
            tensorizing_component()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            tensorizing_pipeline, package_path="tensorizing.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=gcp_project_id, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "engai-tensorizing-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="tensorizing.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCP_SERVICE_ACCOUNT)
    
    if args.age_model_training:
        # Define a Pipeline
        @dsl.pipeline
        def age_model_training_pipeline():
            age_model_training_component()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            age_model_training_pipeline, package_path="age_model_training.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=gcp_project_id, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "engai-age-model-training-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="age_model_training.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCP_SERVICE_ACCOUNT)
    
    if args.model_deployment:
        # Define a Pipeline
        @dsl.pipeline
        def model_deployment_pipeline():
            model_deployment_component()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            model_deployment_pipeline, package_path="model_deployment.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=gcp_project_id, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "engai-age-model-training-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="model_deployment.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCP_SERVICE_ACCOUNT)
    
    if args.pipeline:
        # Define a Pipeline
        @dsl.pipeline
        def ml_pipeline():
            preprocessing_task = data_preprocessing_component().set_display_name('Data Preprocessing')
            tensorizing_task = (
                tensorizing_component()
                .set_display_name('Tensorizing')
                .after(preprocessing_task)
            )
            age_model_training_task = (
                age_model_training_component()
                .set_display_name('Model Training')
                .after(tensorizing_task)
            )
            _ = (
                model_deployment_component()
                .set_display_name('Model Deployment')
                .after(age_model_training_task)
            )
        
        compiler.Compiler().compile(ml_pipeline, package_path='pipeline.yaml')
        
        # Submit job to Vertex AI
        aip.init(project=gcp_project_id, staging_bucket=BUCKET_URI)
        job_id = generate_uuid()
        display_name = "engai-ml-pipeline-" + job_id
        job = aip.PipelineJob(
            display_name=display_name,
            template_path="pipeline.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )
        job.run(service_account=GCP_SERVICE_ACCOUNT)


@dsl.container_component
def data_preprocessing_component():
    container_spec = dsl.ContainerSpec(
        image=DATA_PREPROCESSING_IMAGE,
        command=[],
        args=[],
    )
    return container_spec

@dsl.container_component
def tensorizing_component():
    container_spec = dsl.ContainerSpec(
        image=TENSORIZING_IMAGE,
        command=[],
        args=[],
    )
    return container_spec

@dsl.container_component
def age_model_training_component():
    container_spec = dsl.ContainerSpec(
        image=AGE_MODEL_TRAINING_IMAGE,
        command=[],
        args=[],
    )
    return container_spec

@dsl.container_component
def model_deployment_component():
    container_spec = dsl.ContainerSpec(
        image=MODEL_DEPLOYMENT_IMAGE,
        command=[],
        args=[],
    )
    return container_spec

if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Workflow CLI")

    parser.add_argument(
        "-p",
        "--data_preprocessing",
        action="store_true",
        help="Run just the data preprocessing task in the pipeline.",
    )
    parser.add_argument(
        "-t",
        "--tensorizing",
        action="store_true",
        help="Run just the tensorizing task in the pipeline.",
    )
    parser.add_argument(
        "-a",
        "--age_model_training",
        action="store_true",
        help="Run just the age model training task in the pipeline.",
    )
    parser.add_argument(
        "-d",
        "--model_deployment",
        action="store_true",
        help="Run just the model deployment task in the pipeline.",
    )
    parser.add_argument(
        "-w",
        "--pipeline",
        action="store_true",
        help="Run full pipeline",
    )

    args = parser.parse_args()

    main(args)