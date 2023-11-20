"""
Module that contains the command line app.

Typical usage example from command line:
        python cli.py --upload
        python cli.py --deploy
"""

import os

import argparse
from google.cloud import aiplatform
import tensorflow as tf

# W&B
import wandb

GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_MODELS_BUCKET_NAME = os.environ["GCS_MODELS_BUCKET_NAME"]
BEST_MODEL = "engai/DogNet-breed/DogNet_breed_student_distilled:latest"
ARTIFACT_URI = f"gs://{GCS_MODELS_BUCKET_NAME}/{BEST_MODEL}"


def main(args=None):
    if args.upload:
        print("Upload model to GCS")

        # Download model artifact from wandb
        # run = wandb.init()
        # artifact = run.use_artifact(
        #     BEST_MODEL,
        #     type="model",
        # )
        # artifact_dir = artifact.download()
        # print("artifact_dir", artifact_dir)

        # Load model
        prediction_model = tf.keras.models.load_model("artifacts")
        print(prediction_model.summary())

        # Preprocess Image
        def preprocess_image(bytes_input):
            decoded = tf.io.decode_jpeg(bytes_input, channels=3)
            decoded = tf.image.convert_image_dtype(decoded, tf.float32)
            resized = tf.image.resize(decoded, size=(224, 224))
            return resized

        @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
        def preprocess_function(bytes_inputs):
            decoded_images = tf.map_fn(
                preprocess_image, bytes_inputs, dtype=tf.float32, back_prop=False
            )
            return {"model_input": decoded_images}

        @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
        def serving_function(bytes_inputs):
            images = preprocess_function(bytes_inputs)
            results = model_call(**images)
            return results

        model_call = tf.function(prediction_model.call).get_concrete_function(
            [
                tf.TensorSpec(
                    shape=[None, 224, 224, 3], dtype=tf.float32, name="model_input"
                )
            ]
        )

        # Save updated model to GCS
        tf.saved_model.save(
            prediction_model,
            ARTIFACT_URI,
            signatures={"serving_default": serving_function},
        )

    elif args.deploy:
        print("Deploy model")

        # List of prebuilt containers for prediction
        # https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
        serving_container_image_uri = (
            "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"
        )

        # Upload and Deploy model to Vertex AI
        # Reference: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.Model#google_cloud_aiplatform_Model_upload
        deployed_model = aiplatform.Model.upload(
            display_name=BEST_MODEL,
            artifact_uri=ARTIFACT_URI,
            serving_container_image_uri=serving_container_image_uri,
        )
        print("deployed_model:", deployed_model)
        # Reference: https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.Model#google_cloud_aiplatform_Model_deploy
        endpoint = deployed_model.deploy(
            deployed_model_display_name=BEST_MODEL,
            traffic_split={"0": 100},
            machine_type="n1-standard-4",
            accelerator_count=0,
            min_replica_count=1,
            max_replica_count=1,
            sync=False,
        )
        print("endpoint:", endpoint)


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal 'python cli.py --help', it will provide the description
    parser = argparse.ArgumentParser(description="Data Collector CLI")

    parser.add_argument(
        "-u",
        "--upload",
        action="store_true",
        help="Upload saved model to GCS Bucket",
    )
    parser.add_argument(
        "-d",
        "--deploy",
        action="store_true",
        help="Deploy saved model to Vertex AI",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Test deployment to Vertex AI",
    )

    args = parser.parse_args()

    main(args)
