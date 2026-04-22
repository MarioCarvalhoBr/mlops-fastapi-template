from unittest.mock import patch

from fastapi.testclient import TestClient


class TestSentimentAPI:
    """
    Integration tests for the Sentiment Analysis endpoints.
    Verifies HTTP status codes, Pydantic validations, and routing.
    """

    @patch("src.api.routes.analyze_product_review")
    def test_analyze_review_endpoint_success(self, mock_service, client: TestClient):
        """Test a successful HTTP POST request to the sentiment endpoint."""
        # Mock the service response
        mock_service.return_value = {
            "primary_sentiment": "positive",
            "confidence": 0.99,
            "metadata": {
                "model": "test-model",
                "raw_label": "5 stars",
                "version": "0.2.0"
            }
        }

        payload = {"text": "I absolutely love this new laptop!"}
        response = client.post("/analyze-review", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["primary_sentiment"] == "positive"
        assert "confidence" in data
        assert "metadata" in data

    def test_analyze_review_validation_error_too_short(self, client: TestClient):
        """Test Pydantic min_length validation (Security by Design)."""
        payload = {"text": "ok"} # Less than 3 characters
        response = client.post("/analyze-review", json=payload)

        # 422 Unprocessable Entity is FastAPI's default for Pydantic validation failures
        assert response.status_code == 422 
        assert "detail" in response.json()

    @patch("src.api.routes.analyze_product_review")
    def test_analyze_review_business_logic_error(self, mock_service, client: TestClient):
        """Test how the API handles ValueErrors raised by the service layer."""
        mock_service.side_effect = ValueError("Invalid input detected by service.")

        payload = {"text": "Valid length but rejected by service logic."}
        response = client.post("/analyze-review", json=payload)

        # The route is designed to catch ValueError and return 400 Bad Request
        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid input detected by service."

    @patch("src.api.routes.analyze_product_review")
    def test_analyze_review_internal_server_error(self, mock_service, client: TestClient):
        """Test how the API handles unexpected system crashes."""
        mock_service.side_effect = Exception("Database connection lost.")

        payload = {"text": "This will cause an unexpected crash."}
        response = client.post("/analyze-review", json=payload)

        # The route catches generic Exceptions and returns 500 Internal Server Error
        assert response.status_code == 500