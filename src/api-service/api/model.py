# model.py

import base64
import os
import wandb
from google.cloud import aiplatform
import numpy as np
import tensorflow as tf

class ModelController(object):
    """
    ModelController keeps track of which model is being used for
    DawgAI and is responsible for making prediction requests for
    inference.

    Features:
    * Calls VertexAI endpoint for prediction using remote model.
    * Downloads model from WandB and stores locally for local prediction.
    """

    def __init__(self, model_config, index_to_breed_map) -> None:
        self.use_self_hosted_model = False
        self.project_id = model_config['project_id']
        self.model_id = model_config['model_id']
        self.endpoint_id = model_config['endpoint_id']
        self.location = model_config['location']
        self.wandb_model_name = model_config['wandb_model_name']
        # Defines whether you should use a locally stored model or call VertexAI.
        self.use_local_model = model_config['use_local_model']
        self.index_to_breed_map = index_to_breed_map
        self.local_prediction_model = None
        
        # Initialize an AI Platform client
        aiplatform.init(project=self.project_id, location=self.location)
    
    def predict(self, image: bytes) -> dict:
        """Makes a model prediction request."""
        if self.use_local_model and self.local_prediction_model:
            return self._predict_local(image)
        else:
            return self._predict_remote(image)

    def _predict_remote(self, image: bytes) -> dict:
        """Makes a model prediction using a remote model in VertexAI."""
        # Prepare input and get the model endpoint
        image_bytes = base64.b64encode(image).decode('utf-8')
        instances = [{'bytes_inputs': {'b64': image_bytes}}]
        endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{self.project_id}/locations/{self.location}/endpoints/{self.endpoint_id}")

        # Send a prediction request
        response = endpoint.predict(instances=instances)

        # Find breed from prediction response
        max_probability = max(response.predictions[0])
        max_probability_index = response.predictions[0].index(max_probability)
        predicted_breed = self.index_to_breed_map[str(max_probability_index)]

        return {
            'max_probability': max_probability,
            'max_probability_index': max_probability_index,
            'predicted_breed': predicted_breed,
        }

    def _predict_local(self, image: bytes) -> dict:
        """Makes a model prediction using a locally saved model."""
        model_input = self._preprocess_image(image)
        response = self.local_prediction_model.predict(model_input)
        max_probability = float(max(response[0]))
        max_probability_index = int(np.where(response[0] == max_probability)[0][0])
        predicted_breed = self.index_to_breed_map[str(max_probability_index)]

        return {
            'max_probability': max_probability,
            'max_probability_index': max_probability_index,
            'predicted_breed': predicted_breed,
        }
    
    def download_model_from_wandb(self, wandb_key: str):
        """Downloads a model from W&B and saves it to a local file."""
        wandb.login(key=wandb_key)
        run = wandb.init()
        artifact = run.use_artifact(self.wandb_model_name, type='model')
        artifact_dir = self._get_artifact_dir(artifact)

        prediction_model = tf.keras.models.load_model(artifact_dir)
        print(prediction_model.summary())
        self.local_prediction_model = prediction_model

    def _get_artifact_dir(self, artifact) -> str:
        """
        Only download the model from W&B if it doesn't exist locally.

        Returns the artifact_url if it exists locally, otherwise
        downloads the artifact and returns the provided artifact_url.
        """
        existing_artifact_dir = f'/app/artifacts/{artifact.name.split(":")[0]}:{artifact.version}'
        if os.path.exists(existing_artifact_dir):
            print(f'Existing artifact_dir: {existing_artifact_dir}')
            return existing_artifact_dir

        new_artifact_dir = artifact.download()
        print(f'New artifact_dir: {new_artifact_dir}')
        return new_artifact_dir
    
    def _preprocess_image(self, image):
        """Preprocesses image for prediction using local model."""
        image_width = 224
        image_height = 224
        num_channels = 3

        # Decode & resize
        def decode_resize_image(image):
            image = tf.image.decode_jpeg(image, channels=num_channels)
            image = tf.image.resize(image, [image_height, image_width])
            return image

        # Normalize pixels
        def normalize(image):
            image = image / 255
            return image

        model_input = tf.data.Dataset.from_tensor_slices(([image]))
        model_input = model_input.map(decode_resize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        model_input = model_input.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        model_input = model_input.repeat(1).batch(1)

        return model_input