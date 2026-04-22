import torch

from src.core.logger import logger
from src.models.retrieval_model import RetrievalModel

# Load embedding model singleton
_retrieval_model = RetrievalModel()
logger.info(f"Retrieval Service initialized with model: {_retrieval_model.model_name}")

# In-memory product catalog database
# In production, this would be synchronized with an external Vector DB like Milvus
_product_catalog = [
    {
        "id": "P001",
        "name": "Wireless Noise-Canceling Headphones",
        "category": "Electronics",
        "description": "High-quality over-ear headphones with active noise cancellation and 30-hour battery life. Perfect for travel.",
    },
    {
        "id": "P002",
        "name": "Ergonomic Office Chair",
        "category": "Furniture",
        "description": "Comfortable mesh chair with lumbar support and adjustable armrests to prevent back pain during long working hours.",
    },
    {
        "id": "P003",
        "name": "Mechanical Gaming Keyboard",
        "category": "Gaming",
        "description": "RGB backlit mechanical keyboard with tactile blue switches and an aerospace-grade aluminum frame.",
    },
    {
        "id": "P004",
        "name": "Smartphone Gimbal Stabilizer",
        "category": "Photography",
        "description": "3-axis handheld gimbal for smartphones, perfect for smooth video recording and vlogging on the go.",
    },
    {
        "id": "P005",
        "name": "Stainless Steel Water Bottle",
        "category": "Sports & Outdoors",
        "description": "Insulated double-wall water bottle that keeps drinks cold for 24 hours or hot for 12 hours. Leak-proof.",
    },
    {
        "id": "P006",
        "name": "Running Shoes",
        "category": "Footwear",
        "description": "Lightweight breathable mesh running shoes with advanced shock absorption and a durable rubber outsole.",
    },
]

logger.info(f"Indexing in-memory catalog with {len(_product_catalog)} items...")
# Compute vector embeddings for the entire catalog at startup
_catalog_descriptions = [f"{p['name']} - {p['description']}" for p in _product_catalog]
_product_embeddings = _retrieval_model.encode(_catalog_descriptions)
logger.info("Catalog indexing complete.")


def search_similar_products(query: str, top_k: int = 3) -> dict:
    """
    Executes a semantic search over the in-memory product catalog based on cosine similarity.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    logger.info(f"Executing semantic search for query: '{query}'")

    try:
        # Encode user search intent
        query_embedding = _retrieval_model.encode([query])

        # Compute cosine similarities against the entire catalog tensor
        similarities = torch.nn.functional.cosine_similarity(query_embedding, _product_embeddings)

        # Extract top K matching indices
        top_results = torch.topk(similarities, min(top_k, len(_product_catalog)))

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            idx_val = idx.item()
            product = _product_catalog[idx_val]
            results.append(
                {
                    "id": product["id"],
                    "name": product["name"],
                    "category": product["category"],
                    "description": product["description"],
                    "similarity_score": round(score.item(), 4),
                }
            )

        return {
            "query": query,
            "results": results,
            "metadata": {
                "model": _retrieval_model.model_name,
                "version": _retrieval_model.version,
                "catalog_size": len(_product_catalog),
            },
        }
    except Exception as e:
        logger.error(f"Semantic search failed: {str(e)}")
        raise
