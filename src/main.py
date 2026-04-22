import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config.config import load_config
from src.core.logger import logger

settings = load_config()

app = FastAPI(
    title="Retail-AI MLOps API",
    version="0.2.0",
    description="API de alta performance para inferência de modelos open-source (E-commerce)",
)

# Configuração de CORS para permitir que o frontend comunique com a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, substitua "*" pelo domínio exato do seu frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    logger.info(f"Iniciando servidor em {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )