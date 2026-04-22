from src.core.logger import logger
from src.models.sentiment_model import SentimentModel

_sentiment_model = SentimentModel()


def analyze_product_review(review_text: str) -> dict:
    if not review_text or not review_text.strip():
        raise ValueError("Empty input text.")

    try:
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
