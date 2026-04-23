"""
Retrieval Model Layer.
Contains logic handling vector encoding of texts to dense numerical embeddings using natively implemented inference.
These vectors map objects into an N-dimensional joint-space.
"""
import os

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.config.config import load_config
from src.core.logger import logger

settings = load_config()


class RetrievalModel:
    """
    Abstraction layer for Semantic Search and Vector Embeddings.
    Computes dense vector representations of textual data to calculate relational similarities.
    """

    def __init__(self) -> None:
        """
        Coordinates the initialization of Tokenizer and Transformer architectures.
        Respects device availability constraints like mapping execution layers into accessible resources natively (CPU/GPU).
        """
        self.model_name = settings.retrieval_model_id
        self.version = "0.4.0"

        # Explicitly instruct the core engine to not try reaching out to API endpoints if running isolated environments.
        if settings.hf_hub_offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logger.info("HF Offline Mode enabled for Retrieval Model.")

        logger.info(f"Loading retrieval model {self.model_name} on device: {settings.device}")

        # Load the base model and tokenizer natively. We use Auto instances because it grants
        # granular control of the sequence pool mechanism over `pipeline("feature-extraction")`.
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(settings.device)
        self.device = settings.device

    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        Converts a list of strings into a normalized PyTorch tensor of embeddings.
        This employs the Mean Pooling strategy over hidden states ensuring sequence sizes evaluate correctly regardless of length.

        Args:
            texts (list[str]): The batch of textual content representing descriptions or queries.

        Returns:
            torch.Tensor: L2 normalized numeric representation reflecting the semantic density distributions.
        """
        # Tokenize sentences: padding normalizes vector batches, truncation ensures we don't violate context boundaries.
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        # Compute token embeddings, wrapped in no_grad for significant VRAM/memory reduction (we are only inferring).
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform mean pooling across the output vectors using the attention mask mapping.
        attention_mask = encoded_input["attention_mask"]
        token_embeddings = model_output.last_hidden_state

        # Expand mask dimensions in order properly zero pad the irrelevant values without skewing average computations.
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # L2 Normalize the embeddings. Required for cosine similarity logic applied downstream so directions are preserved uniformly.
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings
