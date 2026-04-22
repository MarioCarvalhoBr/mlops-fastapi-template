import os
from transformers import pipeline
from src.config.config import load_config
from src.core.logger import logger

settings = load_config()

class SentimentModel:
    """
    Abstração do Modelo usando Hugging Face Transformers.
    Focado em execução open-source e suporte a CPU/GPU.
    """

    def __init__(self) -> None:
        self.model_name = settings.sentiment_model_id
        self.version = "0.2.0"
        
        # Injeção de dependência via variável de ambiente para garantir modo offline
        if settings.hf_hub_offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logger.info("Modo Hugging Face Offline ativado.")

        logger.info(f"Carregando modelo {self.model_name} em dispositivo: {settings.device}")
        
        # O pipeline lida com o tokenizador e a inferência automaticamente
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            device=0 if settings.device == "cuda" else -1
        )

    def predict(self, text: str) -> dict:
        """
        Executa a inferência no modelo pré-treinado.
        """
        # O pipeline retorna uma lista de dicionários, pegamos o primeiro resultado
        result = self.classifier(text)[0]
        
        # O modelo nlptown retorna rótulos de "1 star" a "5 stars"
        # Precisamos converter isso para o domínio de negócio do Retail-AI
        label_str = result.get('label', '')
        
        if label_str in ["1 star", "2 stars"]:
            business_label = "negative"
        elif label_str == "3 stars":
            business_label = "neutral"
        elif label_str in ["4 stars", "5 stars"]:
            business_label = "positive"
        else:
            business_label = "unknown"
            
        return {
            "label": business_label,
            "score": round(result['score'], 4),
            "original_output": label_str
        }