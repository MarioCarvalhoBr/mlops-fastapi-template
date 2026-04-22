from unittest.mock import patch

import pytest

from src.services.sentiment_service import analyze_product_review


class TestSentimentService:
    """
    Test suite for the Sentiment Service business logic.
    We mock the underlying model to test the service's orchestration capabilities.
    """

    @patch("src.services.sentiment_service._sentiment_model")
    def test_analyze_product_review_success(self, mock_model):
        """Test successful review analysis formatting and metadata mapping."""
        # Setup mock model behavior
        mock_model.predict.return_value = {"label": "positive", "score": 0.95, "original_output": "4 stars"}
        mock_model.model_name = "mocked-bert-model"
        mock_model.version = "0.2.0"

        # Execute business logic
        response = analyze_product_review("Very good quality.")

        # Assertions
        assert response["primary_sentiment"] == "positive"
        assert response["confidence"] == 0.95
        assert response["metadata"]["model"] == "mocked-bert-model"
        assert response["metadata"]["raw_label"] == "4 stars"
        assert response["metadata"]["version"] == "0.2.0"

    def test_analyze_product_review_empty_input(self):
        """Test that empty strings are rejected by the business layer."""
        with pytest.raises(ValueError, match="Empty input text."):
            analyze_product_review("")

        with pytest.raises(ValueError, match="Empty input text."):
            analyze_product_review("   ")

    @patch("src.services.sentiment_service._sentiment_model")
    def test_analyze_product_review_model_failure(self, mock_model):
        """Test that model exceptions are caught and raised as a specific ValueError."""
        # Simulate a crash in the ML model (e.g., Out Of Memory)
        mock_model.predict.side_effect = RuntimeError("Model OOM Error")

        # The service layer is designed to catch this and raise a ValueError
        with pytest.raises(ValueError, match="Failed to analyze product review sentiment."):
            analyze_product_review("This will crash the mock.")
