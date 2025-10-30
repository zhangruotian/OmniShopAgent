"""
Search Tools for Product Discovery
Provides text-based, image-based, and VLM reasoning capabilities
"""

import base64
import logging
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from openai import OpenAI

from app.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.milvus_service import MilvusService

logger = logging.getLogger(__name__)

# Initialize services as singletons
_embedding_service: Optional[EmbeddingService] = None
_milvus_service: Optional[MilvusService] = None
_openai_client: Optional[OpenAI] = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_milvus_service() -> MilvusService:
    global _milvus_service
    if _milvus_service is None:
        _milvus_service = MilvusService()
        _milvus_service.connect()
    return _milvus_service


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


@tool
def search_products(query: str, limit: int = 5) -> str:
    """Search for fashion products using natural language descriptions.

    Use when users describe what they want:
    - "Find me red summer dresses"
    - "Show me blue running shoes"
    - "I want casual shirts for men"

    Args:
        query: Natural language product description
        limit: Maximum number of results (1-20)

    Returns:
        Formatted string with product information
    """
    try:
        logger.info(f"Searching products: '{query}', limit: {limit}")

        embedding_service = get_embedding_service()
        milvus_service = get_milvus_service()

        if not milvus_service.is_connected():
            milvus_service.connect()

        query_embedding = embedding_service.get_text_embedding(query)

        results = milvus_service.search_similar_text(
            query_embedding=query_embedding,
            limit=min(limit, 20),
            filters=None,
            output_fields=[
                "id",
                "productDisplayName",
                "gender",
                "masterCategory",
                "subCategory",
                "articleType",
                "baseColour",
                "season",
                "usage",
            ],
        )

        if not results:
            return "No products found matching your search."

        output = f"Found {len(results)} product(s):\n\n"

        for idx, product in enumerate(results, 1):
            output += f"{idx}. {product.get('productDisplayName', 'Unknown Product')}\n"
            output += f"   ID: {product.get('id', 'N/A')}\n"
            output += f"   Category: {product.get('masterCategory', 'N/A')} > {product.get('subCategory', 'N/A')} > {product.get('articleType', 'N/A')}\n"
            output += f"   Color: {product.get('baseColour', 'N/A')}\n"
            output += f"   Gender: {product.get('gender', 'N/A')}\n"

            if product.get("season"):
                output += f"   Season: {product.get('season')}\n"
            if product.get("usage"):
                output += f"   Usage: {product.get('usage')}\n"

            if "distance" in product:
                similarity = 1 - product["distance"]
                output += f"   Relevance: {similarity:.2%}\n"

            output += "\n"

        return output.strip()

    except Exception as e:
        logger.error(f"Error searching products: {e}", exc_info=True)
        return f"Error searching products: {str(e)}"


@tool
def search_by_image(image_path: str, limit: int = 5) -> str:
    """Find similar fashion products using an image.

    Use when users want visually similar items:
    - User uploads an image and asks "find similar items"
    - "Show me products that look like this"

    Args:
        image_path: Path to the image file
        limit: Maximum number of results (1-20)

    Returns:
        Formatted string with similar products
    """
    try:
        logger.info(f"Image search: '{image_path}', limit: {limit}")

        img_path = Path(image_path)
        if not img_path.exists():
            return f"Error: Image file not found at '{image_path}'"

        embedding_service = get_embedding_service()
        milvus_service = get_milvus_service()

        if not milvus_service.is_connected():
            milvus_service.connect()

        if (
            not hasattr(embedding_service, "clip_client")
            or embedding_service.clip_client is None
        ):
            embedding_service.connect_clip()

        image_embedding = embedding_service.get_image_embedding(image_path)

        if image_embedding is None:
            return "Error: Failed to generate embedding for image"

        results = milvus_service.search_similar_images(
            query_embedding=image_embedding,
            limit=min(limit + 1, 21),
            filters=None,
            output_fields=[
                "id",
                "image_path",
                "productDisplayName",
                "gender",
                "masterCategory",
                "subCategory",
                "articleType",
                "baseColour",
                "season",
                "usage",
            ],
        )

        if not results:
            return "No similar products found."

        # Filter out the query image itself
        query_id = img_path.stem
        filtered_results = []
        for result in results:
            result_path = result.get("image_path", "")
            if Path(result_path).stem != query_id:
                filtered_results.append(result)
            if len(filtered_results) >= limit:
                break

        if not filtered_results:
            return "No similar products found."

        output = f"Found {len(filtered_results)} visually similar product(s):\n\n"

        for idx, product in enumerate(filtered_results, 1):
            output += f"{idx}. {product.get('productDisplayName', 'Unknown Product')}\n"
            output += f"   ID: {product.get('id', 'N/A')}\n"
            output += f"   Category: {product.get('masterCategory', 'N/A')} > {product.get('subCategory', 'N/A')} > {product.get('articleType', 'N/A')}\n"
            output += f"   Color: {product.get('baseColour', 'N/A')}\n"
            output += f"   Gender: {product.get('gender', 'N/A')}\n"

            if product.get("season"):
                output += f"   Season: {product.get('season')}\n"
            if product.get("usage"):
                output += f"   Usage: {product.get('usage')}\n"

            if "distance" in product:
                similarity = 1 - product["distance"]
                output += f"   Visual Similarity: {similarity:.2%}\n"

            output += "\n"

        return output.strip()

    except Exception as e:
        logger.error(f"Error in image search: {e}", exc_info=True)
        return f"Error searching by image: {str(e)}"


@tool
def analyze_image_style(image_path: str) -> str:
    """Analyze a fashion product image using AI vision to extract detailed style information.

    Use when you need to understand style/attributes from an image:
    - Understand the style, color, pattern of a product
    - Extract attributes like "casual", "formal", "vintage"
    - Get detailed descriptions for subsequent searches

    Args:
        image_path: Path to the image file

    Returns:
        Detailed text description of the product's visual attributes
    """
    try:
        logger.info(f"Analyzing image with VLM: '{image_path}'")

        img_path = Path(image_path)
        if not img_path.exists():
            return f"Error: Image file not found at '{image_path}'"

        with open(img_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        prompt = """Analyze this fashion product image and provide a detailed description.

Include:
- Product type (e.g., shirt, dress, shoes, pants, bag)
- Primary colors
- Style/design (e.g., casual, formal, sporty, vintage, modern)
- Pattern or texture (e.g., plain, striped, checked, floral)
- Key features (e.g., collar type, sleeve length, fit)
- Material appearance (if obvious, e.g., denim, cotton, leather)
- Suitable occasion (e.g., office wear, party, casual, sports)

Provide a comprehensive yet concise description (3-4 sentences)."""

        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
            temperature=0.3,
        )

        analysis = response.choices[0].message.content.strip()
        logger.info("VLM analysis completed")

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing image: {e}", exc_info=True)
        return f"Error analyzing image: {str(e)}"


def get_all_tools():
    """Get all available tools for the agent"""
    return [search_products, search_by_image, analyze_image_style]
