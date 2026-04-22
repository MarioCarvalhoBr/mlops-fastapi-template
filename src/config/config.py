import os
from pathlib import Path
from typing import Any, Dict
import torch
import yaml
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    environment: str = "development"
    log_level: str = "INFO"

    # Sentiment Analysis Model (v0.2.0)
    sentiment_model_id: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    
    # Object Detection Model (v0.3.0)
    vision_model_id: str = "facebook/detr-resnet-50"
    vision_confidence_threshold: float = 0.85
    
    # Hardware and execution mode
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hf_hub_offline: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

def load_yaml_config(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            return config if config is not None else {}
    except (yaml.YAMLError, Exception):
        return {}

def load_config() -> Settings:
    settings = Settings()
    config_path = "configs/config.yaml"
    if os.path.exists(config_path):
        yaml_config = load_yaml_config(config_path)
        for key, value in yaml_config.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
    return settings