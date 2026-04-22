import io
from PIL import Image, UnidentifiedImageError
from src.core.logger import logger
from src.models.object_detector import ObjectDetectorModel
from src.config.config import load_config

settings = load_config()
_object_detector = ObjectDetectorModel()
logger.info(f"Vision Service initialized with model: {_object_detector.model_name}")

def analyze_product_image(image_bytes: bytes) -> dict:
    """
    Validates the image payload, converts it to PIL format, 
    and extracts high-confidence objects.
    """
    if not image_bytes:
        raise ValueError("Empty image data received.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        logger.warning("Invalid image format provided.")
        raise ValueError("The provided file is not a valid image.")
    except Exception as e:
        logger.error(f"Error processing image bytes: {str(e)}")
        raise RuntimeError("Failed to decode the image.")

    logger.info(f"Processing image for object detection. Size: {image.size}")

    try:
        raw_predictions = _object_detector.predict(image)
        
        # Filter predictions based on business confidence threshold
        detected_objects = []
        for pred in raw_predictions:
            if pred["score"] >= settings.vision_confidence_threshold:
                detected_objects.append({
                    "label": pred["label"],
                    "confidence": round(pred["score"], 4),
                    "bounding_box": pred["box"]
                })
        
        # Sort by confidence descending
        detected_objects.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "detected_objects": detected_objects,
            "object_count": len(detected_objects),
            "metadata": {
                "model": _object_detector.model_name,
                "version": _object_detector.version,
                "threshold_applied": settings.vision_confidence_threshold
            }
        }
    except Exception as e:
        logger.error(f"Model inference failed: {str(e)}")
        raise RuntimeError("Object detection inference failed.")