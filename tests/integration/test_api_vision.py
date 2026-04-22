import io
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image

from src.main import app

client = TestClient(app)


@patch("src.api.routes.analyze_product_image")
def test_detect_objects_success(mock_analyze: MagicMock) -> None:
    mock_analyze.return_value = {
        "detected_objects": [{"label": "laptop", "confidence": 0.95, "bounding_box": {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}}],
        "object_count": 1,
        "metadata": {"model": "test", "version": "1.0", "threshold_applied": 0.5},
    }

    img = Image.new("RGB", (10, 10))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")

    response = client.post("/detect-objects", files={"file": ("test.jpg", img_byte_arr.getvalue(), "image/jpeg")})

    assert response.status_code == 200
    assert response.json()["object_count"] == 1
    assert response.json()["detected_objects"][0]["label"] == "laptop"


def test_detect_objects_invalid_type() -> None:
    response = client.post("/detect-objects", files={"file": ("test.txt", b"hello", "text/plain")})

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


@patch("src.api.routes.analyze_product_image")
def test_detect_objects_value_error(mock_analyze: MagicMock) -> None:
    mock_analyze.side_effect = ValueError("Bad image")

    img = Image.new("RGB", (10, 10))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")

    response = client.post("/detect-objects", files={"file": ("test.jpg", img_byte_arr.getvalue(), "image/jpeg")})

    assert response.status_code == 400
    assert response.json()["detail"] == "Bad image"


@patch("src.api.routes.analyze_product_image")
def test_detect_objects_runtime_error(mock_analyze: MagicMock) -> None:
    mock_analyze.side_effect = Exception("Boom")

    img = Image.new("RGB", (10, 10))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")

    response = client.post("/detect-objects", files={"file": ("test.jpg", img_byte_arr.getvalue(), "image/jpeg")})

    assert response.status_code == 500
    assert "Image processing failure" in response.json()["detail"]
