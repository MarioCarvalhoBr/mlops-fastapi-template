import os
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from src.core.logger import logger
from src.models.multimodal_model import MultimodalModel
from src.config.config import load_config

settings = load_config()
_multimodal_model = MultimodalModel()

_catalog_items = []
_catalog_embeddings = None

def _create_dummy_catalog(base_path: Path):
    """Generates a dummy file-based catalog with the standard directory structure."""
    logger.info("Catalog directory not found. Creating dummy catalog...")
    base_path.mkdir(parents=True, exist_ok=True)
    
    dummy_products = [
        {"id": "P001", "name": "Black Backpack", "category": "Accessories", "description": "Durable black travel backpack."},
        {"id": "P002", "name": "Red Running Shoes", "category": "Footwear", "description": "Sportive red shoes for marathon."}
    ]
    
    for prod in dummy_products:
        # Create product root directory
        prod_dir = base_path / prod["id"]
        prod_dir.mkdir(exist_ok=True)
        
        # Create images subdirectory
        images_dir = prod_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Save product metadata
        with open(prod_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump(prod, f, indent=4)
        
        # Generate a dummy image inside the images/ directory
        img = Image.new("RGB", (300, 300), color=(79, 70, 229)) # Indigo background
        d = ImageDraw.Draw(img)
        d.text((20, 140), prod["name"], fill=(255, 255, 255))
        img.save(images_dir / "img_1.jpg")

def load_multimodal_catalog():
    """Loads products and their images from disk, computing joint embeddings."""
    global _catalog_items, _catalog_embeddings
    
    catalog_path = Path(settings.catalog_data_path)
    if not catalog_path.exists():
        _create_dummy_catalog(catalog_path)
    
    items = []
    embeddings_list = []
    
    # Supported modern image extensions
    supported_extensions = [
        "*.jpg", "*.jpeg", "*.png", "*.webp", "*.avif",
        "*.JPG", "*.JPEG", "*.PNG", "*.WEBP", "*.AVIF"
    ]
    
    logger.info(f"Scanning catalog path: {catalog_path}")
    
    for folder in catalog_path.iterdir():
        if folder.is_dir():
            info_file = folder / "info.json"
            images_dir = folder / "images"
            
            if not info_file.exists():
                continue
            
            try:
                # 1. Load product metadata
                with open(info_file, "r", encoding="utf-8") as f:
                    info = json.load(f)
                
                # 2. Encode product textual info
                text_content = f"{info.get('name', '')} {info.get('description', '')}"
                text_emb = _multimodal_model.encode_text(text_content)
                
                # 3. Discover and load images from the images/ subdirectory
                image_embs = []
                image_files = []
                
                if images_dir.exists() and images_dir.is_dir():
                    for ext in supported_extensions:
                        image_files.extend(list(images_dir.glob(ext)))
                
                for img_path in image_files:
                    try:
                        img = Image.open(img_path).convert("RGB")
                        image_embs.append(_multimodal_model.encode_image(img))
                    except Exception as e:
                        logger.warning(f"Failed to load image {img_path}: {e}")
                
                # 4. Late Fusion Strategy: Average text embedding and all valid image embeddings
                combined_emb = text_emb
                if image_embs:
                    avg_img_emb = torch.mean(torch.stack(image_embs), dim=0)
                    combined_emb = (text_emb + avg_img_emb) / 2.0
                
                # 5. Re-normalize the joint embedding
                combined_emb = torch.nn.functional.normalize(combined_emb, p=2, dim=-1)
                
                items.append(info)
                embeddings_list.append(combined_emb)
                
            except Exception as e:
                logger.error(f"Error processing product {folder.name}: {e}")

    _catalog_items = items
    if embeddings_list:
        _catalog_embeddings = torch.cat(embeddings_list, dim=0)
    
    logger.info(f"Multimodal catalog loaded successfully: {len(_catalog_items)} products indexed.")

# Execute initialization load
load_multimodal_catalog()

def search_multimodal_catalog(query_text: str = None, query_image_bytes: bytes = None, top_k: int = 3) -> dict:
    """
    Executes a multimodal search combining text intent and visual features.
    """
    if not _catalog_items or _catalog_embeddings is None:
        raise RuntimeError("Catalog is empty or failed to load.")
    
    if not query_text and not query_image_bytes:
        raise ValueError("Must provide either a text query or an image file.")

    logger.info(f"Executing multimodal search. Text: {bool(query_text)}. Image: {bool(query_image_bytes)}")

    try:
        query_embs = []
        
        # Process text query
        if query_text:
            query_embs.append(_multimodal_model.encode_text(query_text))
            
        # Process image query
        if query_image_bytes:
            img = Image.open(io.BytesIO(query_image_bytes)).convert("RGB")
            query_embs.append(_multimodal_model.encode_image(img))
            
        # Combine inputs via mean pooling if both are provided
        if len(query_embs) > 1:
            joint_query_emb = torch.mean(torch.stack(query_embs), dim=0)
        else:
            joint_query_emb = query_embs[0]
            
        joint_query_emb = torch.nn.functional.normalize(joint_query_emb, p=2, dim=-1)
        
        # Compute cosine similarities in the joint vector space
        similarities = torch.nn.functional.cosine_similarity(joint_query_emb, _catalog_embeddings)
        
        # Retrieve top K results
        top_results = torch.topk(similarities, min(top_k, len(_catalog_items)))
        
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            idx_val = idx.item()
            product = _catalog_items[idx_val]
            results.append({
                "id": product.get("id", "N/A"),
                "name": product.get("name", "Unknown"),
                "category": product.get("category", "Unknown"),
                "description": product.get("description", ""),
                "similarity_score": round(score.item(), 4)
            })
            
        query_type = "text+image" if query_text and query_image_bytes else "text" if query_text else "image"
            
        return {
            "query_type": query_type,
            "results": results,
            "metadata": {
                "model": _multimodal_model.model_name,
                "version": _multimodal_model.version,
                "catalog_size": len(_catalog_items)
            }
        }
    except Exception as e:
        logger.error(f"Multimodal search failed: {str(e)}")
        raise