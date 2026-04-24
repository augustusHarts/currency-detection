from fastapi import HTTPException
from app.utils.config import MODEL_URL
from fastapi import Request

class Inference:
    _model = None

    def _get_model(self):
        if Inference._model is None:
            # Import here to avoid heavy startup imports (Render port binding).
            from ultralytics import YOLO
            Inference._model = YOLO(str(MODEL_URL))
        return Inference._model

    def model_prediction(
        self, 
        image: str
    ):
        try:
            model = self._get_model()
            results = model.predict(
                source=image,
                conf=0.4,
                iou=0.45,
                device='cpu',
                imgsz=416,
                task='detect',
                verbose=False
            )
            return results[0]
        except Exception as e:
            import traceback
            traceback.print_exc() # This prints the REAL error to your console
            raise HTTPException(status_code=500, detail=str(e))
        
        
