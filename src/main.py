import uvicorn
from fastapi import FastAPI

from src.api.routes import router
from src.config.config import load_config
from src.core.logger import logger

settings = load_config()

app = FastAPI(
    title="MLOps API",
    version="1.0.0",
    description="API for serving Machine Learning models",
)

app.include_router(router)

if __name__ == "__main__":
    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
