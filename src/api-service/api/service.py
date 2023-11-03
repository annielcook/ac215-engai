# service.py

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

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