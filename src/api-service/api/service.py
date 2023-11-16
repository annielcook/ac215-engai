# service.py

import json
import os
from fastapi import FastAPI, File, Form
from starlette.middleware.cors import CORSMiddleware
from tempfile import TemporaryDirectory
from .model import ModelController

app = FastAPI(title='EngAI API Server', description='API Server', version='v1')

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event('startup')
async def startup():
    print('Server starting up...')
    with open('config/model-controller-config.json', 'r') as f:
        model_config = json.load(f)
    with open('config/index-to-breed.json', 'r') as f:
        index_to_breed_map = json.load(f)
    with open('../secrets/wandb.json', 'r') as f:
        wandb_key = json.load(f)['wandb_key']

    global model_controller
    model_controller = ModelController(model_config, index_to_breed_map)
    if model_config['use_local_model']:
        model_controller.download_model_from_wandb(wandb_key)



# Routes
@app.get('/')
async def get_index():
    return {'message': 'This is the EngAI API service.'}

@app.post('/predict')
async def predict(image: bytes = File(...), file_type: str = Form(...)):
    print("image:", len(image), type(image))
    print("image type:", file_type)
    
    response = model_controller.predict(image)
    print(response)
    return response