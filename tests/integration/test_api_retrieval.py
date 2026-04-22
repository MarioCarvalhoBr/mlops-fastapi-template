from unittest.mock import patch

from fastapi.testclient import TestClient


class TestRetrievalAPI:
    """
    Integration tests for the Semantic Search / Retrieval endpoints.
    Verifies HTTP status codes, Pydantic validations, and routing.
    """

    @patch("src.api.routes.search_similar_products")
    def test_semantic_search_endpoint_success(self, mock_service, client: TestClient):
        """Test a successful HTTP POST request to the semantic search endpoint."""
        # Mock the service response
        mock_service.return_value = {
            "query": "headphones",
            "results": [
                {
                    "id": "P001",
                    "name": "Wireless Noise-Canceling Headphones",
                    "category": "Electronics",
                    "description": "High-quality over-ear headphones.",
                    "similarity_score": 0.985,
                }
            ],
            "metadata": {"model": "sentence-transformers/all-MiniLM-L6-v2", "version": "0.4.0", "catalog_size": 1},
        }

        payload = {"query": "headphones", "top_k": 1}
        response = client.post("/semantic-search", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "headphones"
        assert len(data["results"]) == 1
        assert data["results"][0]["id"] == "P001"
        assert data["metadata"]["catalog_size"] == 1

    def test_semantic_search_validation_error_too_short(self, client: TestClient):
        """Test Pydantic min_length validation for the query."""
        payload = {"query": "a", "top_k": 3}  # Less than 2 characters
        response = client.post("/semantic-search", json=payload)

        # 422 Unprocessable Entity
        assert response.status_code == 422
        assert "detail" in response.json()

    def test_semantic_search_validation_error_top_k_bounds(self, client: TestClient):
        """Test Pydantic range validation for top_k."""
        # Test lower bound
        payload_low = {"query": "headphones", "top_k": 0}
        response_low = client.post("/semantic-search", json=payload_low)
        assert response_low.status_code == 422

        # Test upper bound
        payload_high = {"query": "headphones", "top_k": 15}
        response_high = client.post("/semantic-search", json=payload_high)
        assert response_high.status_code == 422

    @patch("src.api.routes.search_similar_products")
    def test_semantic_search_business_logic_error(self, mock_service, client: TestClient):
        """Test how the API handles ValueErrors raised by the service layer."""
        mock_service.side_effect = ValueError("Search query cannot be empty.")

        payload = {"query": "   ", "top_k": 3}
        response = client.post("/semantic-search", json=payload)

        # 400 Bad Request
        assert response.status_code == 400
        assert response.json()["detail"] == "Search query cannot be empty."

    @patch("src.api.routes.search_similar_products")
    def test_semantic_search_internal_server_error(self, mock_service, client: TestClient):
        """Test how the API handles unexpected system crashes."""
        mock_service.side_effect = Exception("Out of Memory")

        payload = {"query": "Find me something that crashes", "top_k": 3}
        response = client.post("/semantic-search", json=payload)

        # 500 Internal Server Error
        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to process semantic search."
