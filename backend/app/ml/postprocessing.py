from PIL import Image as PILImage
from app.utils.config import OUTPUT_DIR
import numpy as np
import cv2

class PostProcessing:
    def __init__(self):
        pass

    def draw_detections(
        self, 
        image_path: str, 
        output_path: str,
        result
    ) -> str:

        img = cv2.imread(image_path)
        if img is None: return ""

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf, cls = float(box.conf[0]), int(box.cls[0])
                label = f"{result.names[cls]} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 0), 2)
                cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 0), 2)

        cv2.imwrite(output_path, img)
        return output_path