"""
Sentiment Service Layer.
Manages the sentiment analysis workflow by validating inputs and coordinating
with the underlying sentiment model to extract sentiment from customer interactions
such as product reviews.
"""
from src.core.logger import logger
from src.models.sentiment_model import SentimentModel

# Singleton instantiation to ensure the heavy ML model is loaded only once at startup
_sentiment_model = SentimentModel()


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
        # Delegate pure prediction task to the model
        prediction = _sentiment_model.predict(review_text)
        return {
            "primary_sentiment": prediction["label"],
            "confidence": prediction["score"],
            "metadata": {
                "model": _sentiment_model.model_name,
                "raw_label": prediction["original_output"],
                "version": _sentiment_model.version,
            },
        }
    except Exception as e:
        logger.error(f"Error in sentiment service: {str(e)}")
        raise ValueError("Failed to analyze product review sentiment.")
