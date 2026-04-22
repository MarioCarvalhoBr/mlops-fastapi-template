import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.config.config import load_config
from src.core.logger import logger

settings = load_config()


class MultimodalModel:
    """
    Multimodal Zero-Shot Retrieval Model using CLIP.
    Maps text and images to the same joint embedding space.
    """

    def __init__(self) -> None:
        self.model_name = settings.multimodal_model_id
        self.version = "0.5.0"
        self.device = settings.device

        if settings.hf_hub_offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logger.info("HF Offline Mode enabled for Multimodal Model.")

        logger.info(f"Loading multimodal model {self.model_name} on device: {self.device}")

        # Load CLIP architecture
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Encodes text into the multimodal vector space."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        text_features = self.model.get_text_features(**inputs)
        # Normalize the embeddings for cosine similarity
        return F.normalize(text_features, p=2, dim=-1)

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encodes an image into the multimodal vector space."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        return F.normalize(image_features, p=2, dim=-1)
