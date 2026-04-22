from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from src.models.multimodal_model import MultimodalModel


class TestMultimodalModel:
    """
    Test suite for the MultimodalModel abstraction.
    We mock Hugging Face classes to avoid download and GPU needs.
    """

    @patch("src.models.multimodal_model.CLIPModel.from_pretrained")
    @patch("src.models.multimodal_model.CLIPProcessor.from_pretrained")
    def test_model_initialization(self, mock_processor: MagicMock, mock_model: MagicMock) -> None:
        """Test if the multimodal model initializes correctly."""
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance

        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        model = MultimodalModel()

        mock_processor.assert_called_once()
        mock_model.assert_called_once()
        assert model.version == "0.5.0"
        assert model.processor == mock_processor_instance
        assert model.model == mock_model_instance

    @patch("src.models.multimodal_model.CLIPModel.from_pretrained")
    @patch("src.models.multimodal_model.CLIPProcessor.from_pretrained")
    def test_encode_text_success(self, mock_processor: MagicMock, mock_model: MagicMock) -> None:
        """Test encoding text into a joint vector space."""
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        mock_text_features = torch.ones((1, 4))
        mock_model_instance.get_text_features.return_value = mock_text_features

        mock_processor_instance = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_processor_instance.return_value = mock_inputs
        mock_processor.return_value = mock_processor_instance

        model = MultimodalModel()
        model.device = "cpu"

        text = "sample multimodal query"
        embeddings = model.encode_text(text)

        mock_processor_instance.assert_called_once_with(
            text=[text], return_tensors="pt", padding=True, truncation=True
        )
        mock_model_instance.get_text_features.assert_called_once()

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (1, 4)
        norm = torch.linalg.norm(embeddings, dim=-1)
        assert pytest.approx(norm.item()) == 1.0

    @patch("src.models.multimodal_model.CLIPModel.from_pretrained")
    @patch("src.models.multimodal_model.CLIPProcessor.from_pretrained")
    def test_encode_image_success(self, mock_processor: MagicMock, mock_model: MagicMock) -> None:
        """Test encoding image into a joint vector space."""
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        mock_image_features = torch.ones((1, 4))
        mock_model_instance.get_image_features.return_value = mock_image_features

        mock_processor_instance = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = {"pixel_values": torch.zeros((1, 3, 224, 224))}
        mock_processor_instance.return_value = mock_inputs
        mock_processor.return_value = mock_processor_instance

        model = MultimodalModel()
        model.device = "cpu"

        img = Image.new("RGB", (224, 224), color="red")
        embeddings = model.encode_image(img)

        mock_processor_instance.assert_called_once_with(images=img, return_tensors="pt")
        mock_model_instance.get_image_features.assert_called_once()

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (1, 4)
        norm = torch.linalg.norm(embeddings, dim=-1)
        assert pytest.approx(norm.item()) == 1.0
