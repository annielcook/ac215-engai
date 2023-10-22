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

DATA_PREPROCESSING_IMAGE = "abzp/ac215-model-training:abpujare"


def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def main(args=None):
    print("CLI Arguments:", args)
    with open(GOOGLE_APP_CREDENTIALS, 'r') as f:
        secrets = json.load(f)
    
    gcs_service_account = secrets['private_key_id']
    gcp_project_id = secrets['project_id']

    if args.data_preprocessing:
        # Define a Container Component
        @dsl.container_component
        def data_preprocessing_component():
            container_spec = dsl.ContainerSpec(
                image=DATA_PREPROCESSING_IMAGE,
                command=[],
                args=[],
            )
            return container_spec

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

        job.run(service_account='32226619505-compute@developer.gserviceaccount.com')

if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Workflow CLI")

    parser.add_argument(
        "-p",
        "--data_preprocessing",
        action="store_true",
        help="Run just the Data preprocessing",
    )
    parser.add_argument(
        "-w",
        "--pipeline",
        action="store_true",
        help="Run full pipeline",
    )

    args = parser.parse_args()

    main(args)