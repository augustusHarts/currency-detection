from app.ml.preprocessing import PreProcessing
from app.ml.postprocessing import PostProcessing
from app.ml.inference import Inference
from app.utils.config import INPUT_DIR, OUTPUT_DIR, VALUE_MAP, MIME_TYPES

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Request
from pathlib import Path
from collections import Counter

import os
import uuid
import cv2
import shutil
import numpy as np

router = APIRouter(prefix='/v1',tags=['Items'])

@router.post('/predict')
async def predict(
    request: Request,
    file: UploadFile = File(...),
    conf: float = 0.5    
):
    preprocess = request.app.state.preprocess
    postprocess = request.app.state.postprocess
    inference = request.app.state.inference

    conf = max(0.05, min(0.95, float(conf)))
    
    if not file.content_type or not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f'Expects an image file.'
        )

    if file.content_type not in MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 
            detail=f"Unsupported file type: {file.content_type}. Please upload a JPEG, PNG, or WebP image."
        )

    content = await file.read()
    
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large")

    # await file.seek(0)

    unique_id = str(uuid.uuid4())
    input_path = str(INPUT_DIR / f"{unique_id}_{file.filename}")
    no_bg_path = str(INPUT_DIR / f"nobg_{unique_id}.png")
    final_output_path = str(OUTPUT_DIR / f"result_{unique_id}.jpg")

    try:
        with open(input_path, "wb") as buffer:
            buffer.write(content)

        preprocess.remove_background_white(Path(input_path), Path(no_bg_path))
        result = inference.model_prediction(no_bg_path)

        postprocess.draw_detections(no_bg_path, final_output_path, result)

        detections = []
        total_amount = 0
        label_counts = Counter()

        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                score = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                det_obj = {
                    "label": label,
                    "confidence": round(score, 4),
                    "box": {
                        "x1": round(xyxy[0], 2), "y1": round(xyxy[1], 2),
                        "x2": round(xyxy[2], 2), "y2": round(xyxy[3], 2),
                    }
                }
                detections.append(det_obj)
                
                if label in VALUE_MAP:
                    total_amount += VALUE_MAP[label]
                    label_counts[label] += 1    

        breakdown = []
        for lbl in sorted(label_counts, key=lambda k: VALUE_MAP.get(k, 0), reverse=True):
            val = VALUE_MAP[lbl]
            cnt = label_counts[lbl]
            breakdown.append({"denomination": val, "count": cnt, "subtotal": val * cnt})

        return {
            "image_width": result.orig_shape[1],
            "image_height": result.orig_shape[0],
            "detections": detections,
            "total_amount": total_amount,
            "breakdown": breakdown,
            "currency": "INR",
            # Absolute URL so the frontend can always load it
            "processed_image_url": str(
                request.url_for("output", path=os.path.basename(final_output_path))
            ),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")