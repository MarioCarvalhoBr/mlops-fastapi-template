from unittest.mock import MagicMock, patch

from PIL import Image

from src.models.object_detector import ObjectDetectorModel


@patch("src.models.object_detector.pipeline")
@patch("src.models.object_detector.settings")
def test_object_detector_init(mock_settings: MagicMock, mock_pipeline: MagicMock) -> None:
    mock_settings.vision_model_id = "test-model"
    mock_settings.device = "cpu"
    mock_settings.hf_hub_offline = False

    detector = ObjectDetectorModel()

    mock_pipeline.assert_called_once_with("object-detection", model="test-model", device=-1)
    assert detector.model_name == "test-model"


@patch("src.models.object_detector.pipeline")
@patch("src.models.object_detector.settings")
def test_predict(mock_settings: MagicMock, mock_pipeline: MagicMock) -> None:
    mock_settings.vision_model_id = "test-model"
    mock_settings.device = "cpu"

    mock_pipeline_instance = MagicMock()
    mock_pipeline.return_value = mock_pipeline_instance
    mock_pipeline_instance.return_value = [{"label": "cat", "score": 0.99, "box": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}}]

    detector = ObjectDetectorModel()
    img = Image.new("RGB", (100, 100))
    result = detector.predict(img)

    mock_pipeline_instance.assert_called_once_with(img)
    assert len(result) == 1
    assert result[0]["label"] == "cat"
