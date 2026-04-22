import io
from unittest.mock import patch

import pytest
import torch
from PIL import Image

# We will need to repatch internal variables for testing since they are module-level
import src.services.multimodal_service as multimodal_service
from src.services.multimodal_service import search_multimodal_catalog


class TestMultimodalService:
    @pytest.fixture(autouse=True)
    def setup_mock_catalog(self):
        """Setup fake catalog items and embeddings for multimodal service."""
        # Save originals
        self.original_items = multimodal_service._catalog_items
        self.original_embeddings = multimodal_service._catalog_embeddings

        # Create mock data
        multimodal_service._catalog_items = [
            {"id": "P001", "name": "Backpack", "category": "Bag", "description": "A bag"},
            {"id": "P002", "name": "Shoes", "category": "Shoes", "description": "For running"},
        ]
        # Two embeddings of size 4
        multimodal_service._catalog_embeddings = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

        yield

        # Restore originals
        multimodal_service._catalog_items = self.original_items
        multimodal_service._catalog_embeddings = self.original_embeddings

    @patch.object(multimodal_service._multimodal_model, "encode_text")
    def test_search_multimodal_text_only(self, mock_encode_text):
        """Test multimodal search using only text intent."""
        # Mock encoding to have identical semantic feature as index 1 (Shoes)
        mock_encode_text.return_value = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

        result = search_multimodal_catalog(query_text="Running Shoes", top_k=1)

        assert result["query_type"] == "text"
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "P002"
        assert result["results"][0]["similarity_score"] == pytest.approx(1.0)
        mock_encode_text.assert_called_once_with("Running Shoes")

    @patch.object(multimodal_service._multimodal_model, "encode_image")
    def test_search_multimodal_image_only(self, mock_encode_image):
        """Test multimodal search using only an image."""
        # Predict backpack
        mock_encode_image.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        img = Image.new("RGB", (10, 10), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        image_bytes = img_bytes.getvalue()

        result = search_multimodal_catalog(query_image_bytes=image_bytes, top_k=1)

        assert result["query_type"] == "image"
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "P001"
        assert result["results"][0]["similarity_score"] == pytest.approx(1.0)
        mock_encode_image.assert_called_once()

    @patch.object(multimodal_service._multimodal_model, "encode_text")
    @patch.object(multimodal_service._multimodal_model, "encode_image")
    def test_search_multimodal_late_fusion(self, mock_encode_image, mock_encode_text):
        """Test multimodal search with both text and image (late fusion)."""
        mock_encode_text.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        mock_encode_image.return_value = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

        img = Image.new("RGB", (10, 10), color="green")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        image_bytes = img_bytes.getvalue()

        result = search_multimodal_catalog(query_text="Backpack and Shoes", query_image_bytes=image_bytes, top_k=2)

        assert result["query_type"] == "text+image"
        # The joint embedding is mean([1,0,0,0], [0,1,0,0]) = [0.5, 0.5, 0, 0]
        # Normalized: [0.707, 0.707, 0, 0]
        # Similarity with [1,0,0,0] is 0.707, and with [0,1,0,0] is 0.707
        assert len(result["results"]) == 2
        assert result["results"][0]["similarity_score"] == pytest.approx(0.7071, rel=1e-3)
        assert result["results"][1]["similarity_score"] == pytest.approx(0.7071, rel=1e-3)
        mock_encode_text.assert_called_once_with("Backpack and Shoes")
        mock_encode_image.assert_called_once()

    def test_search_multimodal_empty_inputs(self):
        """Test validation when neither text nor image is provided."""
        with pytest.raises(ValueError, match="Must provide either a text query or an image file."):
            search_multimodal_catalog()

    def test_search_multimodal_empty_catalog(self):
        """Test search behavior when catalog is empty."""
        multimodal_service._catalog_embeddings = None
        multimodal_service._catalog_items = []

        with pytest.raises(RuntimeError, match="Catalog is empty or failed to load."):
            search_multimodal_catalog(query_text="Dummy")
