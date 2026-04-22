from fastapi import APIRouter, File, HTTPException, UploadFile, Form
from pydantic import BaseModel, Field

from typing import Optional


from src.core.logger import logger
from src.services.multimodal_service import search_multimodal_catalog
from src.services.retrieval_service import search_similar_products
from src.services.sentiment_service import analyze_product_review
from src.services.vision_service import analyze_product_image

router = APIRouter()

# --- Schemas ---
class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000)

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500, description="Semantic search query intent.")
    top_k: int = Field(3, ge=1, le=10, description="Number of top results to retrieve.")

class SentimentMetadata(BaseModel):
    model: str
    raw_label: str
    version: str

class SentimentResponse(BaseModel):
    primary_sentiment: str
    confidence: float
    metadata: SentimentMetadata

class DetectedObject(BaseModel):
    label: str
    confidence: float
    bounding_box: dict

class VisionMetadata(BaseModel):
    model: str
    version: str
    threshold_applied: float

class VisionResponse(BaseModel):
    detected_objects: list[DetectedObject]
    object_count: int
    metadata: VisionMetadata

class CatalogProduct(BaseModel):
    id: str
    name: str
    category: str
    description: str
    similarity_score: float

class RetrievalMetadata(BaseModel):
    model: str
    version: str
    catalog_size: int

class SearchResponse(BaseModel):
    query: str
    results: list[CatalogProduct]
    metadata: RetrievalMetadata
    

class MultimodalResponse(BaseModel):
    query_type: str
    results: list[CatalogProduct]
    metadata: dict

# --- Endpoints ---
@router.get("/health")
async def health_check():
    """Service health and version check."""
    return {"status": "ok", "version": "0.4.0"}

@router.post("/analyze-review", response_model=SentimentResponse, tags=["Retail-AI Sentiment"])
async def analyze_review_route(request: ReviewRequest):
    try:
        return analyze_product_review(request.text)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="Sentiment analysis failure.")

@router.post("/detect-objects", response_model=VisionResponse, tags=["Retail-AI Vision"])
async def detect_objects_route(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be an image.")
    try:
        image_bytes = await file.read()
        return analyze_product_image(image_bytes)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in vision endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Image processing failure.")

@router.post("/semantic-search", response_model=SearchResponse, tags=["Retail-AI Retrieval"])
async def semantic_search_route(request: SearchRequest):
    """
    Semantic Search endpoint mapping user intent to product vector embeddings.
    """
    try:
        return search_similar_products(request.query, request.top_k)
    except ValueError as ve:
        logger.warning(f"Search validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Retrieval inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process semantic search.")
    
    
# --- NOVA ROTA V0.5.0 ---
@router.post("/multimodal-search", response_model=MultimodalResponse, tags=["Retail-AI Multimodal"])
async def multimodal_search_route(
    query: Optional[str] = Form(None, description="Optional text query"),
    file: Optional[UploadFile] = File(None, description="Optional image file"),
    top_k: int = Form(3)
):
    """
    Zero-Shot Multimodal Search endpoint mapping text and/or images to a joint vector space.
    """
    if not query and not file:
        raise HTTPException(status_code=400, detail="Provide either a text query or an image file.")

    image_bytes = None
    if file:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image.")
        image_bytes = await file.read()

    try:
        return search_multimodal_catalog(query, image_bytes, top_k)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Multimodal inference error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process multimodal search.")