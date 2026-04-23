"""
Object Detection Model Layer.
Provides a clean abstraction over the Hugging Face Transformers pipeline for object detection.
Isolates deep learning specific logic, device management, and pipeline setup from the rest of the app.
"""
import os

from PIL import Image
from transformers import pipeline

from src.config.config import load_config
from src.core.logger import logger

settings = load_config()


class ObjectDetectorModel:
    """
    Object detection model abstraction using Hugging Face Transformers.
    Utilizes DETR (End-to-End Object Detection) to identify items in images.
    """

    def __init__(self) -> None:
        """
        Initializes the model pipeline based on application settings.
        Handles offline mode configurations and optimal device allocation (CPU vs GPU).
        """
        self.model_name = settings.vision_model_id
        self.version = "0.3.0"

        # Force the library to use only locally cached models if offline mode is toggled via settings.
        if settings.hf_hub_offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logger.info("Hugging Face Offline Mode is enabled.")

        logger.info(f"Loading vision model {self.model_name} on device: {settings.device}")

        # Initialize the pipeline, moving execution to the GPU (device=0) if "cuda" is available.
        self.detector = pipeline(
            "object-detection",
            model=self.model_name,
            device=0 if settings.device == "cuda" else -1,
        )

    def predict(self, image: Image.Image) -> list:
        """
        Executes object detection on a given PIL Image.

        Args:
            image (Image.Image): The preprocessed RGB image.

        Returns:
            list: A list of detection dictionaries, each containing 'label', 'score', and 'box'.
        """
        # The transformers pipeline handles the batching and tensor conversions natively.
        results = self.detector(image)
        return results
