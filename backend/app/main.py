from fastapi import FastAPI
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
from app.apis.image_route import router
from app.ml.inference import Inference
from app.ml.postprocessing import PostProcessing
from app.ml.preprocessing import PreProcessing

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing components...")

    app.state.preprocess = PreProcessing()
    app.state.postprocess = PostProcessing()
    app.state.inference = Inference()
    app.state.inference._get_model()

    print("All components loaded")

    yield 

    print("Shutting down...")

app = FastAPI(
    title='Currency Detection Backend',
    version='1.0.0',
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(router, prefix='/api')

@app.get('/health', tags=['Health'])
def health_check():
    return {'status': 'ok'}
