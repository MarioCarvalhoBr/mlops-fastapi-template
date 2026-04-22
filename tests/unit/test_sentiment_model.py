from unittest.mock import MagicMock, patch

import pytest

from src.models.sentiment_model import SentimentModel


class TestSentimentModel:
    """
    Test suite for the SentimentModel abstraction.
    We mock the Hugging Face pipeline to ensure tests are fast, deterministic,
    and do not require internet access or GPU hardware.
    """

    @patch("src.models.sentiment_model.pipeline")
    def test_model_initialization(self, mock_pipeline):
        """Test if the model initializes the HF pipeline correctly."""
        model = SentimentModel()
        
        mock_pipeline.assert_called_once()
        assert model.model_name == "nlptown/bert-base-multilingual-uncased-sentiment"
        assert model.version == "0.2.0"
        assert hasattr(model, "classifier")

    @patch("src.models.sentiment_model.pipeline")
    def test_predict_positive_sentiment(self, mock_pipeline):
        """Test mapping of 4-5 stars to positive sentiment."""
        # Setup the mock to return a simulated Hugging Face output
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{"label": "5 stars", "score": 0.98}]
        mock_pipeline.return_value = mock_classifier

        model = SentimentModel()
        result = model.predict("Absolutely fantastic product!")

        assert result["label"] == "positive"
        assert result["score"] == 0.98
        assert result["original_output"] == "5 stars"

    @patch("src.models.sentiment_model.pipeline")
    def test_predict_negative_sentiment(self, mock_pipeline):
        """Test mapping of 1-2 stars to negative sentiment."""
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{"label": "1 star", "score": 0.85}]
        mock_pipeline.return_value = mock_classifier

        model = SentimentModel()
        result = model.predict("Terrible experience, broke on day one.")

        assert result["label"] == "negative"
        assert result["score"] == 0.85

    @patch("src.models.sentiment_model.pipeline")
    def test_predict_neutral_sentiment(self, mock_pipeline):
        """Test mapping of 3 stars to neutral sentiment."""
        mock_classifier = MagicMock()
        mock_classifier.return_value = [{"label": "3 stars", "score": 0.50}]
        mock_pipeline.return_value = mock_classifier

        model = SentimentModel()
        result = model.predict("It is okay, nothing special.")

        assert result["label"] == "neutral"