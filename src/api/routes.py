from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from src.services.sentiment_service import analyze_product_review

router = APIRouter()

class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000)

class SentimentResponse(BaseModel):
    primary_sentiment: str
    confidence: float
    metadata: dict

@router.get("/health")
async def health_check():
    return {"status": "ok", "version": "0.2.0"}

@router.post("/analyze-review", response_model=SentimentResponse, tags=["Retail-AI"])
async def analyze_review_route(request: ReviewRequest):
    try:
        return analyze_product_review(request.text)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="Internal inference error.")