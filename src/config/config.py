import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    environment: str = "development"
    log_level: str = "INFO"

    model_path: str = "models/checkpoint.pt"
    enable_cache: bool = True
    cache_ttl: int = 300

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
    except yaml.YAMLError as e:
        print(f"Error loading YAML from {path}: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error loading {path}: {e}")
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
