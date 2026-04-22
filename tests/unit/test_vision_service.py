import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.services.vision_service import analyze_product_image


@patch("src.services.vision_service._object_detector")
@patch("src.services.vision_service.settings")
def test_analyze_product_image_success(mock_settings: MagicMock, mock_detector: MagicMock) -> None:
    mock_settings.vision_confidence_threshold = 0.5

    mock_detector.model_name = "test-model"
    mock_detector.version = "1.0.0"
    mock_detector.predict.return_value = [
        {"label": "cat", "score": 0.9, "box": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}},
        {"label": "dog", "score": 0.3, "box": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}},  # Below threshold
    ]

    img = Image.new("RGB", (10, 10))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    result = analyze_product_image(img_bytes)

    assert result["object_count"] == 1
    assert result["detected_objects"][0]["label"] == "cat"
    assert result["detected_objects"][0]["confidence"] == 0.9
    assert result["metadata"]["model"] == "test-model"


def test_analyze_product_image_empty() -> None:
    with pytest.raises(ValueError, match="Empty image data received."):
        analyze_product_image(b"")


def test_analyze_product_image_invalid() -> None:
    with pytest.raises(ValueError, match="The provided file is not a valid image."):
        analyze_product_image(b"not an image")


@patch("src.services.vision_service._object_detector")
def test_analyze_product_image_model_failure(mock_detector: MagicMock) -> None:
    mock_detector.predict.side_effect = Exception("Model exploded!")

    img = Image.new("RGB", (10, 10))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    with pytest.raises(RuntimeError, match="Object detection inference failed."):
        analyze_product_image(img_bytes)
