# model.py

import base64
from google.cloud import aiplatform

class ModelController(object):
    """
    ModelController keeps track of which model is being used for
    DawgAI and is responsible for making prediction requests for
    inference.

    Features:
    * Calls VertexAI endpoint for prediction using remote model.
    * (TODO) Downloads model from WandB and stores locally for local prediction.
    """

    def __init__(self, model_config, index_to_breed_map) -> None:
        self.use_self_hosted_model = False
        self.project_id = model_config['project_id']
        self.model_id = model_config['model_id']
        self.endpoint_id = model_config['endpoint_id']
        self.location = model_config['location']
        self.index_to_breed_map = index_to_breed_map
        
        # Initialize an AI Platform client
        aiplatform.init(project=self.project_id, location=self.location)
    
    def predict(self, image) -> dict:
        """Reads an image from an image_path and makes a model prediction request."""
        # Get the model and endpoint
        image_bytes = base64.b64encode(image).decode('utf-8')

        endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{self.project_id}/locations/{self.location}/endpoints/{self.endpoint_id}")
        instances = [{'bytes_inputs': {'b64': image_bytes}}]

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
