import os
from transformers import pipeline
from PIL import Image
from src.config.config import load_config
from src.core.logger import logger

settings = load_config()

class ObjectDetectorModel:
    """
    Object detection model abstraction using Hugging Face Transformers.
    Utilizes DETR (End-to-End Object Detection) to identify items in images.
    """

    def __init__(self) -> None:
        self.model_name = settings.vision_model_id
        self.version = "0.3.0"
        
        if settings.hf_hub_offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logger.info("Hugging Face Offline Mode is enabled.")

        logger.info(f"Loading vision model {self.model_name} on device: {settings.device}")
        
        self.detector = pipeline(
            "object-detection",
            model=self.model_name,
            device=0 if settings.device == "cuda" else -1
        )

    def predict(self, image: Image.Image) -> list:
        """
        Executes object detection on a given PIL Image.
        Returns a list of dictionaries containing labels, scores, and bounding boxes.
        """
        results = self.detector(image)
        return results