from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from src.core.logger import logger
from src.services.sentiment_service import analyze_product_review
from src.services.vision_service import analyze_product_image

router = APIRouter()

# --- SCHEMAS ---
class ReviewRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=3, 
        max_length=2000, 
        description="Customer review text. Limited to 2000 chars to prevent DoS."
    )

class SentimentMetadata(BaseModel):
    model: str
    raw_label: str
    version: str

class SentimentResponse(BaseModel):
    primary_sentiment: str
    confidence: float
    metadata: SentimentMetadata

class VisionMetadata(BaseModel):
    model: str
    version: str
    threshold_applied: float

class DetectedObject(BaseModel):
    label: str
    confidence: float
    bounding_box: dict

class VisionResponse(BaseModel):
    detected_objects: list[DetectedObject]
    object_count: int
    metadata: VisionMetadata

# --- ROUTES ---
@router.get("/health")
async def health_check():
    """Health check endpoint to monitor API uptime."""
    return {"status": "ok", "version": "0.3.0"}

@router.post("/analyze-review", response_model=SentimentResponse, tags=["Retail-AI Sentiment"])
async def analyze_review_route(request: ReviewRequest):
    """
    Analyzes the sentiment of a product review using a Hugging Face model.
    """
    try:
        return analyze_product_review(request.text)
    except ValueError as ve:
        logger.warning(f"Bad Request: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to infer sentiment.")

@router.post("/detect-objects", response_model=VisionResponse, tags=["Retail-AI Vision"])
async def detect_objects_route(file: UploadFile = File(...)):
    """
    Upload an image to detect objects and product categories.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        image_bytes = await file.read()
        return analyze_product_image(image_bytes)
    except ValueError as ve:
        logger.warning(f"Image validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Vision inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process image.")