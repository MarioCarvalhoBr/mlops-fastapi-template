"""
API routing layer for the Retail-AI application.
This module defines the HTTP endpoints, input validation models using Pydantic,
and orchestrates calls to the underlying AI services. It maps HTTP requests to
business logic and handles response structuring and error handling.
"""
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.core.logger import logger
from src.services.inference_service import predict_service
from src.services.multimodal_service import search_multimodal_catalog
from src.services.retrieval_service import search_similar_products
from src.services.sentiment_service import analyze_product_review
from src.services.vision_service import analyze_product_image

router = APIRouter()


# --- Schemas ---
# Schemas strictly handle request/response validation and structure.
class ReviewRequest(BaseModel):
    """Schema for validating incoming text reviews."""

    text: str = Field(..., min_length=3, max_length=2000)


class SearchRequest(BaseModel):
    """Schema for validating semantic search requests."""

    query: str = Field(..., min_length=2, max_length=500, description="Semantic search query intent.")
    top_k: int = Field(3, ge=1, le=10, description="Number of top results to retrieve.")


class SentimentMetadata(BaseModel):
    """Metadata regarding the sentiment analysis execution."""

    model: str
    raw_label: str
    version: str


class SentimentResponse(BaseModel):
    """Structured response for sentiment analysis."""

    primary_sentiment: str
    confidence: float
    metadata: SentimentMetadata


class DetectedObject(BaseModel):
    """Represents a single object detected in an image."""

    label: str
    confidence: float
    bounding_box: dict


class VisionMetadata(BaseModel):
    """Metadata regarding the vision analysis execution."""

    model: str
    version: str
    threshold_applied: float


class VisionResponse(BaseModel):
    """Structured response for vision-based object detection."""

    detected_objects: list[DetectedObject]
    object_count: int
    metadata: VisionMetadata


class CatalogProduct(BaseModel):
    """Represents a product retrieved from the catalog."""

    id: str
    name: str
    category: str
    description: str
    similarity_score: float


class RetrievalMetadata(BaseModel):
    """Metadata regarding the semantic search execution."""

    model: str
    version: str
    catalog_size: int


class SearchResponse(BaseModel):
    """Structured response for semantic search containing matching products."""

    query: str
    results: list[CatalogProduct]
    metadata: RetrievalMetadata


class MultimodalResponse(BaseModel):
    """Structured response for multimodal (text + image) search."""

    query_type: str
    results: list[CatalogProduct]
    metadata: dict


# --- Endpoints ---
@router.get("/health")
async def health_check():
    """
    Service health and version check.
    Used by load balancers and orchestrators to verify service availability.
    """
    import torch
    device = "gpu" if torch.cuda.is_available() else "cpu"
    return {"status": "ok", "version": "0.1.0", "system": {"device": device}}


@router.post("/predict")
async def predict_route(input: str):
    """
    Endpoint that receives text via query parameter.
    Delegates to the prediction service.
    """
    prediction_result = predict_service(input)
    return {"prediction": prediction_result, "version": "0.1.0"}


@router.post("/analyze-review", response_model=SentimentResponse, tags=["Retail-AI Sentiment"])
async def analyze_review_route(request: ReviewRequest):
    """
    Receives a product review and returns its sentiment classification.
    Catches business logic exceptions and maps them to 400 Bad Request,
    while mapping unexpected crashes to 500 Internal Server Error.
    """
    try:
        return analyze_product_review(request.text)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=500, detail="Sentiment analysis failure.")


@router.post("/detect-objects", response_model=VisionResponse, tags=["Retail-AI Vision"])
async def detect_objects_route(file: UploadFile = File(...)):
    """
    Receives an image payload, validates the MIME type, and performs object detection.
    Returns bounding boxes and confidence scores for identified items.
    """
    # Enforce basic pre-validation on content type before reading payload into memory
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
    Retrieves the top K catalog items that semantically match the text query.
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
@router.post(
    "/multimodal-search",
    response_model=MultimodalResponse,
    tags=["Retail-AI Multimodal"],
)
async def multimodal_search_route(
    query: Optional[str] = Form(None, description="Optional text query"),
    file: Optional[UploadFile] = File(None, description="Optional image file"),
    top_k: int = Form(3),
):
    """
    Zero-Shot Multimodal Search endpoint mapping text and/or images to a joint vector space.
    Accepts text, image, or both to locate relevant items in the catalog.
    """
    # Require at least one input modality to perform a valid search
    if not query and not file:
        raise HTTPException(status_code=400, detail="Provide either a text query or an image file.")

    image_bytes = None
    if file:
        # Validate that the file is indeed an image before loading bytes
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
