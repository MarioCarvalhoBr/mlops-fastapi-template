"""
Vision Service Layer.
Orchestrates object detection workflows. It handles data conversion from raw HTTP payloads
to image formats compatible with the Hugging Face models, executes the model predictions,
and applies business logic thresholds to filter results.
"""
import io

from PIL import Image, UnidentifiedImageError

from src.config.config import load_config
from src.core.logger import logger
from src.models.object_detector import ObjectDetectorModel

settings = load_config()

# Instantiate the object detector model uniquely as a singleton to avoid reloading into memory per request.
_object_detector = ObjectDetectorModel()
logger.info(f"Vision Service initialized with model: {_object_detector.model_name}")


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
        # Perform pure inference via the model abstraction class
        raw_predictions = _object_detector.predict(image)

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
                "model": _object_detector.model_name,
                "version": _object_detector.version,
                "threshold_applied": settings.vision_confidence_threshold,
            },
        }
    except Exception as e:
        logger.error(f"Model inference failed: {str(e)}")
        raise RuntimeError("Object detection inference failed.")
