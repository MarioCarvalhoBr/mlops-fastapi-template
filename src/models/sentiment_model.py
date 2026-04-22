import os
from transformers import pipeline
from src.config.config import load_config
from src.core.logger import logger

settings = load_config()

class SentimentModel:
    def __init__(self) -> None:
        self.model_name = settings.sentiment_model_id
        self.version = "0.2.0"
        
        if settings.hf_hub_offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        logger.info(f"Carregando {self.model_name} em {settings.device}")
        
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            device=0 if settings.device == "cuda" else -1
        )

    def predict(self, text: str) -> dict:
        result = self.classifier(text)[0]
        label_str = result.get('label', '')
        
        # Mapeamento para domínio de negócio
        mapping = {
            "1 star": "negative", "2 stars": "negative",
            "3 stars": "neutral",
            "4 stars": "positive", "5 stars": "positive"
        }
        
        return {
            "label": mapping.get(label_str, "unknown"),
            "score": round(result['score'], 4),
            "original_output": label_str
        }