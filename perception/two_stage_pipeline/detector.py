from ultralytics import YOLO
import numpy as np

class Detector:
    def __init__(self, weights: str = 'yolov8n.pt', device: str = None):
        self.model = YOLO(weights)

    def predict(self, source, imgsz=640, conf=0.25):
        """Run detection on source (file, folder or image). Returns list of results.

        Each result contains `.orig_img` (numpy HWC) and `.boxes` with xyxy tensor.
        """
        results = self.model(source, imgsz=imgsz, conf=conf)
        return results
