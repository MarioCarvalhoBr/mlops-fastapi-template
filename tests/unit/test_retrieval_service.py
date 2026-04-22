from unittest.mock import MagicMock, patch

import pytest
import torch

from src.services.retrieval_service import search_similar_products


class TestRetrievalService:
    """
    Test suite for the Retrieval Service business logic.
    We mock the underlying model and the in-memory catalog DB to test the search behavior natively.
    """

    @patch("src.services.retrieval_service._retrieval_model")
    @patch("src.services.retrieval_service._product_embeddings")
    @patch("src.services.retrieval_service._product_catalog")
    def test_search_similar_products_success(self, mock_catalog, mock_embeddings, mock_model):
        """Test retrieving results with successful cosine similarity logic."""
        # Setup mock model behavior
        mock_model.model_name = "mocked-retrieval-model"
        mock_model.version = "0.4.0"

        # When the search asks for query_embedding, return [1,0,0,0] (100% matched to object 0)
        mock_model.encode.return_value = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        # Simulated embeddings for search querying
        fake_catalog_embeds = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])  # Item 0 matches perfectly  # Item 1 is orthogonal

        # For mocking module level variables that are directly accessed, patch passes them as args,
        # but because they are module-level variables used directly in functions, we must overwrite them in the test:
        import src.services.retrieval_service as rs

        old_model = rs._retrieval_model
        old_embeddings = rs._product_embeddings
        old_catalog = rs._product_catalog

        rs._retrieval_model = mock_model
        rs._product_embeddings = fake_catalog_embeds
        rs._product_catalog = [
            {
                "id": "P001",
                "name": "Headphones",
                "category": "Electronics",
                "description": "Noise-Canceling",
            },
            {
                "id": "P002",
                "name": "Office Chair",
                "category": "Furniture",
                "description": "Ergonomic mesh",
            },
        ]

        try:
            response = search_similar_products("Find me headphones", top_k=1)

            # Checking payload shape
            assert response["query"] == "Find me headphones"
            assert len(response["results"]) == 1

            top_hit = response["results"][0]
            # Should be item P001 because we mocked the query vector to perfectly match it
            assert top_hit["id"] == "P001"
            assert top_hit["name"] == "Headphones"

            # Similarities: dot product [1,0,0,0] * [1,0,0,0] = 1.0 score
            assert response["results"][0]["similarity_score"] == pytest.approx(1.0)

            # Metadata check
            assert response["metadata"]["model"] == "mocked-retrieval-model"
            assert response["metadata"]["catalog_size"] == 2

        finally:
            rs._retrieval_model = old_model
            rs._product_embeddings = old_embeddings
            rs._product_catalog = old_catalog

    def test_search_similar_products_empty_query(self):
        """Test empty queries are rejected."""
        with pytest.raises(ValueError, match="Search query cannot be empty."):
            search_similar_products("")

        with pytest.raises(ValueError, match="Search query cannot be empty."):
            search_similar_products("   ")

    def test_search_similar_products_exception_handling(self):
        """Test standard exception handling for inference failures."""
        import src.services.retrieval_service as rs

        old_model = rs._retrieval_model

        fail_model = MagicMock()
        fail_model.encode.side_effect = RuntimeError("GPU Out of Memory")

        rs._retrieval_model = fail_model

        try:
            with pytest.raises(RuntimeError, match="GPU Out of Memory"):
                search_similar_products("crash this")
        finally:
            rs._retrieval_model = old_model
