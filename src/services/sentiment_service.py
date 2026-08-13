"""
Sentiment Service Layer.
Manages the sentiment analysis workflow by validating inputs and coordinating
with the underlying sentiment model to extract sentiment from customer interactions
such as product reviews.
"""
import threading
from typing import Optional

from src.core.logger import logger
from src.models.sentiment_model import SentimentModel

# Singleton instantiation to ensure the heavy ML model is loaded only once, on the
# first request instead of at import time, keeping server startup instantaneous.
_sentiment_model: Optional[SentimentModel] = None
_load_lock = threading.Lock()


def _get_model() -> SentimentModel:
    """Loads the sentiment model once and reuses it across requests."""
    global _sentiment_model

    # Double-checked locking: requests arriving together during the cold start wait for
    # a single load instead of each pulling its own copy of the weights.
    if _sentiment_model is None:
        with _load_lock:
            if _sentiment_model is None:
                model = SentimentModel()
                logger.info(f"Sentiment Service initialized with model: {model.model_name}")
                _sentiment_model = model

    return _sentiment_model


def analyze_product_review(review_text: str) -> dict:
    """
    Validates a product review text and maps the raw machine-learning output
    to a standardized business sentiment classification.

    Args:
        review_text (str): The raw text review submitted by a user.

    Returns:
        dict: A dictionary carrying the mapped primary sentiment, the model's confidence, and execution metadata.

    Raises:
        ValueError: If input is empty or if execution fails for any reason during model inference.
    """
    # Prevent inference on empty strings to save compute cycles
    if not review_text or not review_text.strip():
        raise ValueError("Empty input text.")

    try:
        sentiment_model = _get_model()

        # Delegate pure prediction task to the model
        prediction = sentiment_model.predict(review_text)
        return {
            "primary_sentiment": prediction["label"],
            "confidence": prediction["score"],
            "metadata": {
                "model": sentiment_model.model_name,
                "raw_label": prediction["original_output"],
                "version": sentiment_model.version,
            },
        }
    except Exception as e:
        logger.error(f"Error in sentiment service: {str(e)}")
        raise ValueError("Failed to analyze product review sentiment.")
