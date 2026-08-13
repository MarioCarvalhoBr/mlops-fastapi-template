"""
Vision Service Layer.
Orchestrates object detection workflows. It handles data conversion from raw HTTP payloads
to image formats compatible with the Hugging Face models, executes the model predictions,
and applies business logic thresholds to filter results.
"""
import io
import threading
from typing import Optional

from PIL import Image, UnidentifiedImageError

from src.config.config import load_config
from src.core.logger import logger
from src.models.object_detector import ObjectDetectorModel

settings = load_config()

# Detector singleton, resolved on first use rather than at import time so the API
# (and the static frontend) becomes reachable without waiting on model weights.
_object_detector: Optional[ObjectDetectorModel] = None
_load_lock = threading.Lock()


def _get_detector() -> ObjectDetectorModel:
    """Loads the object detection model once and reuses it across requests."""
    global _object_detector

    # Double-checked locking: requests arriving together during the cold start wait for
    # a single load instead of each pulling its own copy of the weights.
    if _object_detector is None:
        with _load_lock:
            if _object_detector is None:
                detector = ObjectDetectorModel()
                logger.info(f"Vision Service initialized with model: {detector.model_name}")
                _object_detector = detector

    return _object_detector


def analyze_product_image(image_bytes: bytes) -> dict:
    """
    Validates the input image payload, converts it to PIL format,
    runs the detection model, and extracts objects meeting the required confidence score.

    Args:
        image_bytes (bytes): The raw image payload uploaded by the client.

    Returns:
        dict: A structured dictionary containing a list of detected objects, their count, and metadata.

    Raises:
        ValueError: If the image data is empty or corrupted/invalid format.
        RuntimeError: If there is an issue decoding the image or executing the inference.
    """
    if not image_bytes:
        raise ValueError("Empty image data received.")

    try:
        # Load byte payload into a memory buffer and convert to an RGB PIL Image matching the model's expectations.
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        logger.warning("Invalid image format provided.")
        raise ValueError("The provided file is not a valid image.")
    except Exception as e:
        logger.error(f"Error processing image bytes: {str(e)}")
        raise RuntimeError("Failed to decode the image.")

    logger.info(f"Processing image for object detection. Size: {image.size}")

    try:
        object_detector = _get_detector()

        # Perform pure inference via the model abstraction class
        raw_predictions = object_detector.predict(image)

        # Filter predictions based on business confidence threshold
        detected_objects = []
        for pred in raw_predictions:
            # We only keep predictions that the model is highly confident about to avoid false positives.
            if pred["score"] >= settings.vision_confidence_threshold:
                detected_objects.append(
                    {
                        "label": pred["label"],
                        "confidence": round(pred["score"], 4),
                        "bounding_box": pred["box"],
                    }
                )

        # Sort by confidence descending to show the most relevant items first.
        detected_objects.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "detected_objects": detected_objects,
            "object_count": len(detected_objects),
            "metadata": {
                "model": object_detector.model_name,
                "version": object_detector.version,
                "threshold_applied": settings.vision_confidence_threshold,
            },
        }
    except Exception as e:
        logger.error(f"Model inference failed: {str(e)}")
        raise RuntimeError("Object detection inference failed.")
