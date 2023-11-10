# service.py

import os
from fastapi import FastAPI, File, Form
from starlette.middleware.cors import CORSMiddleware
from tempfile import TemporaryDirectory

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

# Routes
@app.get('/')
async def get_index():
    return {'message': 'This is the EngAI API service.'}

@app.post('/predict')
async def predict(image: bytes = File(...), file_type: str = Form(...)):
    print("image:", len(image), type(image))
    print("image type:", file_type)

    with TemporaryDirectory() as image_dir:
        image_path = os.path.join(image_dir, f'predict_image.{file_type}')
        with open(image_path, "wb") as output:
            output.write(image)

    return {'image_path': image_path}