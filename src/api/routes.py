from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.core.logger import logger
from src.services.inference_service import predict_service
from src.services.sentiment_service import analyze_product_review

router = APIRouter()

# --- SCHEMAS DE VALIDAÇÃO (SECURITY BY DESIGN) ---
class ReviewRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=3, 
        max_length=2000, 
        description="Texto da avaliação do cliente. Máximo de 2000 caracteres para evitar DDoS/OOM."
    )

class SentimentResponseMetadata(BaseModel):
    model: str
    raw_label: str
    version: str

class SentimentResponse(BaseModel):
    primary_sentiment: str
    confidence: float
    metadata: SentimentResponseMetadata

# --- ROTAS ---
@router.get("/health")
async def health_check():
    """Endpoint para monitoramento de vivacidade."""
    return {"status": "ok"}

@router.post("/predict")
async def predict_route(input: str):
    """Rota legada (v0.1.0)."""
    prediction_result = predict_service(input)
    return {"prediction": prediction_result}

@router.post("/analyze-review", response_model=SentimentResponse, tags=["Retail-AI"])
async def analyze_review_route(request: ReviewRequest):
    """
    Analisa o sentimento de uma avaliação usando modelo Hugging Face (v0.2.0).
    """
    try:
        result = analyze_product_review(request.text)
        return result
    except ValueError as ve:
        logger.warning(f"Bad Request: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Falha ao inferir sentimento do produto.")