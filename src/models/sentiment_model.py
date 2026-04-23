"""
Sentiment Model Layer.
Provides an integration with Hugging Face text classification pipelines.
Abstracts the complexities regarding device management, model loading, and caching for sentiment analysis tasks.
"""
import os

from transformers import pipeline

from src.config.config import load_config
from src.core.logger import logger

settings = load_config()


class SentimentModel:
    """
    Text classification encapsulation handling natural language review interpretation.
    It maps generic text classification into specific business insights (positive, negative, neutral).
    """

    def __init__(self) -> None:
        """
        Initializes the model pipelines based on configuration flags.
        Respects device availability strategies (CPU/GPU) and hub offline restrictions.
        """
        self.model_name = settings.sentiment_model_id
        self.version = "0.2.0"

        # Apply environment variables if network isolation is requested by infrastructure configuration.
        if settings.hf_hub_offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        logger.info(f"Loading {self.model_name} on {settings.device}")

        # Spin up pipeline explicitly passing the device for PyTorch backend mapping.
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            device=0 if settings.device == "cuda" else -1,
        )  # type: ignore

    def predict(self, text: str) -> dict:
        """
        Calculates sentiment classification score over a given input review text.

        Args:
            text (str): The customer review string.

        Returns:
            dict: An object containing normalized labels, confidence score, and original prediction output.
        """
        # We only pass a single text block, so we take the first element from the returned array.
        result = self.classifier(text)[0]
        label_str = result.get("label", "")

        # Mapping to business domain to normalize the outputs from discrete star rankings to broader semantic meanings.
        mapping = {
            "1 star": "negative",
            "2 stars": "negative",
            "3 stars": "neutral",
            "4 stars": "positive",
            "5 stars": "positive",
        }

        return {
            "label": mapping.get(label_str, "unknown"),
            "score": round(result["score"], 4),
            "original_output": label_str,
        }
