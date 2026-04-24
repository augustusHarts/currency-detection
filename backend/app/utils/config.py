from pathlib import Path

BASE_URL=Path(__file__).resolve().parent.parent

MODEL_URL=BASE_URL / 'models' / 'yolov8n.pt'

INPUT_DIR = BASE_URL / 'data' / 'input'
INPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = BASE_URL / 'data' / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALUE_MAP={
    "10": 10,
    "20": 20,
    "50": 50,
    "100": 100, 
    "200": 200,
    "500": 500,
}

# More robust MIME type validation
MIME_TYPES = [
    "image/jpeg", 
    "image/png", 
    "image/jpg", 
    "image/webp"
]