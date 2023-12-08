"""
Module that contains the command line app.
 
Typical usage example from command line:
        python cli.py
"""

import os
import argparse
import random
import string
from kfp import dsl
from kfp import compiler
import google.cloud.aiplatform as aip
from model import model_training, model_deploy


GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
BUCKET_URI = f"gs://{os.environ['BUCKET_URI']}"
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline_root/root"
GCS_SERVICE_ACCOUNT = os.environ["GCS_SERVICE_ACCOUNT"]
GCP_REGION = os.environ["GCP_REGION"]

# Read the docker tag file
with open(".docker-tag") as f:
    tag = f.read()

tag = tag.strip()

print("Tag>>", tag, "<<")

DATA_PROCESSOR_IMAGE = f"gcr.io/{GCP_PROJECT}/engai-app-data-preprocessing:{tag}"


def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def main(args=None):
    print("CLI Arguments:", args)

    if args.data_preprocessing:
        print("Data Preprocessing")

        # Define a Container Component for data preprocessing
        @dsl.container_component
        def data_preprocessing():
            container_spec = dsl.ContainerSpec(
                image=DATA_PROCESSOR_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--clean",
                    "--prepare",
                    f"--bucket {GCS_BUCKET_NAME}",
                ],
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def data_preprocessing_pipeline():
            data_preprocessing()

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            data_preprocessing_pipeline, package_path="data_preprocessing.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "engai-data-preprocessing-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="data_preprocessing.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.model_training:
        print("Model Training")

        # Define a Pipeline
        @dsl.pipeline
        def model_training_pipeline():
            model_training(
                project=GCP_PROJECT,
                location=GCP_REGION,
                staging_bucket=BUCKET_URI,
                bucket_name=GCS_BUCKET_NAME,
            )

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            model_training_pipeline, package_path="model_training.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "dawg-ai-app-model-training-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="model_training.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.model_deploy:
        print("Model Deploy")

        # Define a Pipeline
        @dsl.pipeline
        def model_deploy_pipeline():
            model_deploy(
                bucket_name=GCS_BUCKET_NAME,
            )

        # Build yaml file for pipeline
        compiler.Compiler().compile(
            model_deploy_pipeline, package_path="model_deploy.yaml"
        )

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "dawg-ai-app-model-deploy-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="model_deploy.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)

    if args.pipeline:
        print("Pipeline")
        
        # Define a Container Component for data preprocessing
        @dsl.container_component
        def data_preprocessing():
            container_spec = dsl.ContainerSpec(
                image=DATA_PROCESSOR_IMAGE,
                command=[],
                args=[
                    "cli.py",
                    "--clean",
                    "--prepare",
                    f"--bucket {GCS_BUCKET_NAME}",
                ],
            )
            return container_spec

        # Define a Pipeline
        @dsl.pipeline
        def ml_pipeline():
            # Data Processor
            data_preprocessing_task = (
                data_preprocessing()
                .set_display_name("Data Processor")
            )
            # Model Training
            model_training_task = (
                model_training(
                    project=GCP_PROJECT,
                    location=GCP_REGION,
                    staging_bucket=BUCKET_URI,
                    bucket_name=GCS_BUCKET_NAME,
                    epochs=1,
                    batch_size=16,
                    model_name="mobilenetv2",
                    train_base=False,
                )
                .set_display_name("Model Training")
                .after(data_preprocessing_task)
            )
            # Model Deployment
            model_deploy_task = (
                model_deploy(
                    bucket_name=GCS_BUCKET_NAME,
                )
                .set_display_name("Model Deploy")
                .after(model_training_task)
            )

        # Build yaml file for pipeline
        compiler.Compiler().compile(ml_pipeline, package_path="pipeline.yaml")

        # Submit job to Vertex AI
        aip.init(project=GCP_PROJECT, staging_bucket=BUCKET_URI)

        job_id = generate_uuid()
        DISPLAY_NAME = "dawg-ai-app-pipeline-" + job_id
        job = aip.PipelineJob(
            display_name=DISPLAY_NAME,
            template_path="pipeline.yaml",
            pipeline_root=PIPELINE_ROOT,
            enable_caching=False,
        )

        job.run(service_account=GCS_SERVICE_ACCOUNT)


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Workflow CLI")

    parser.add_argument(
        "-p",
        "--data_preprocessing",
        action="store_true",
        help="Run just the Data Preprocessing",
    )
    parser.add_argument(
        "-t",
        "--model_training",
        action="store_true",
        help="Run just Model Training",
    )
    parser.add_argument(
        "-d",
        "--model_deploy",
        action="store_true",
        help="Run just Model Deployment",
    )
    parser.add_argument(
        "-w",
        "--pipeline",
        action="store_true",
        help="DawgAI App Pipeline",
    )

    args = parser.parse_args()

    main(args)