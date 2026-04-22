import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from src.config.config import load_config
from src.core.logger import logger

settings = load_config()

class RetrievalModel:
    """
    Abstraction layer for Semantic Search and Vector Embeddings.
    Computes dense vector representations of textual data.
    """

    def __init__(self) -> None:
        self.model_name = settings.retrieval_model_id
        self.version = "0.4.0"
        
        if settings.hf_hub_offline:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logger.info("HF Offline Mode enabled for Retrieval Model.")

        logger.info(f"Loading retrieval model {self.model_name} on device: {settings.device}")
        
        # Load the base model and tokenizer natively to avoid extra dependencies
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(settings.device)
        self.device = settings.device

    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        Converts a list of strings into a normalized PyTorch tensor of embeddings
        using the Mean Pooling strategy over the hidden states.
        """
        # Tokenize sentences
        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(self.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform mean pooling
        attention_mask = encoded_input['attention_mask']
        token_embeddings = model_output.last_hidden_state
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # L2 Normalize the embeddings for cosine similarity processing
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings