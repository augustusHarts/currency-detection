from fastapi import FastAPI
from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.apis.image_route import router
from app.ml.inference import Inference
from app.ml.postprocessing import PostProcessing
from app.ml.preprocessing import PreProcessing
from app.utils.config import OUTPUT_DIR

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing components...")

    app.state.preprocess = PreProcessing()
    app.state.postprocess = PostProcessing()
    app.state.inference = Inference()

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
    allow_origins=['https://currency-detection-eight.vercel.app'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(router, prefix='/api')

# Serve processed images for the frontend to display
app.mount(
    "/data/output",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="output",
)

@app.get("/")
async def root():
    return {"message": "Currency Detection API is running"}

@app.get('/health', tags=['Health'])
def health_check():
    return {'status': 'ok'}
