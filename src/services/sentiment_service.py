from src.core.logger import logger
from src.models.sentiment_model import SentimentModel

# Carregamento da instância em memória na inicialização do serviço
_sentiment_model = SentimentModel()
logger.info(f"Serviço de Análise inicializado com modelo: {_sentiment_model.model_name}")

def analyze_product_review(review_text: str) -> dict:
    """
    Higieniza a entrada e orquestra a predição.
    """
    # Validação de negócio (redundante à API, mas garante a integridade da camada Service)
    if not review_text or not review_text.strip():
        raise ValueError("O texto da avaliação não pode estar vazio.")

    logger.info(f"Analisando avaliação de {len(review_text)} caracteres.")

    try:
        prediction = _sentiment_model.predict(review_text)
        
        return {
            "primary_sentiment": prediction["label"],
            "confidence": prediction["score"],
            "metadata": {
                "model": _sentiment_model.model_name,
                "raw_label": prediction["original_output"],
                "version": _sentiment_model.version
            }
        }
    except Exception as e:
        logger.error(f"Erro na inferência do modelo: {str(e)}")
        raise RuntimeError("Ocorreu um erro ao processar a avaliação. Tente novamente mais tarde.")