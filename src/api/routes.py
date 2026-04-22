from fastapi import APIRouter

from src.core.logger import logger
from src.services.inference_service import predict_service

router = APIRouter()


@router.get("/health")
async def health_check():
    """Endpoint to check if the API is alive."""
    logger.info("Health check requested")
    return {"status": "ok"}


@router.post("/predict")
async def predict_route(input: str):
    """
    Endpoint that receives text via query parameter.
    Delegates to the prediction service.
    """
    prediction_result = predict_service(input)
    return {"prediction": prediction_result}
