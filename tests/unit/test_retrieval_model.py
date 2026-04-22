from unittest.mock import MagicMock, patch

import pytest
import torch

from src.models.retrieval_model import RetrievalModel


class TestRetrievalModel:
    """
    Test suite for the RetrievalModel abstraction.
    We mock the Hugging Face classes to ensure tests are fast, deterministic,
    and do not require internet access or GPU hardware.
    """

    @patch("src.models.retrieval_model.AutoModel.from_pretrained")
    @patch("src.models.retrieval_model.AutoTokenizer.from_pretrained")
    def test_model_initialization(self, mock_tokenizer, mock_model):
        """Test if the retrieval model initializes the base HF classes correctly."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        model = RetrievalModel()

        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()
        assert model.version == "0.4.0"
        assert model.tokenizer == mock_tokenizer_instance
        assert model.model == mock_model_instance

    @patch("src.models.retrieval_model.AutoModel.from_pretrained")
    @patch("src.models.retrieval_model.AutoTokenizer.from_pretrained")
    def test_encode_success(self, mock_tokenizer, mock_model):
        """Test encoding texts into vectors with mean pooling."""
        # Fake device assignment for model
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        # Model output fake (last_hidden_state needs to be 3D tensor: batch, seq_len, hidden_size)
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.ones((2, 3, 4))  # shape: 2 sequences, 3 tokens, 4 embedding dim
        mock_model_instance.return_value = mock_output

        # Tokenizer returns dictionary of tensors with attention mask
        mock_encoded_input = {"input_ids": torch.tensor([[1, 2, 3], [1, 2, 0]]), "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]])}
        # Mock .to(device) for encoded input
        mock_tokenized = MagicMock()
        mock_tokenized.to.return_value = mock_encoded_input

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = mock_tokenized
        mock_tokenizer.return_value = mock_tokenizer_instance

        model = RetrievalModel()
        model.device = "cpu"  # Override to ensure no external dependency

        texts = ["First text", "Second text"]
        embeddings = model.encode(texts)

        # Check tokenization args
        mock_tokenizer_instance.assert_called_once_with(texts, padding=True, truncation=True, return_tensors="pt")

        # Assert returned embeddings are 2D tensor (batch, hidden_size)
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (2, 4)

        # Norms should be 1.0 (since L2 normalization is applied)
        norms = torch.linalg.norm(embeddings, dim=1)
        assert pytest.approx(norms[0].item()) == 1.0
        assert pytest.approx(norms[1].item()) == 1.0
