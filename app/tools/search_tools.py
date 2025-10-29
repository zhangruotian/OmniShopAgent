"""
Search Tools for Product Discovery
Provides text-based, image-based, and VLM reasoning capabilities
"""

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.tools import BaseTool
from openai import OpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.milvus_service import MilvusService

logger = logging.getLogger(__name__)


class TextSearchInput(BaseModel):
    """Input schema for text-based product search"""

    query: str = Field(
        description="Natural language description of the product to search for, e.g., 'red summer dress' or 'blue running shoes'"
    )
    limit: int = Field(
        default=5,
        description="Maximum number of products to return (1-20)",
        ge=1,
        le=20,
    )
    filters: Optional[str] = Field(
        default=None,
        description="Optional filter expression, e.g., 'gender == \"Women\"' or 'baseColour == \"Red\"'",
    )


class ProductSearchTool(BaseTool):
    """Tool for searching products using natural language descriptions"""

    name: str = "search_products"
    description: str = """
    Search for fashion products using natural language descriptions.
    
    Use this tool when users ask to find products by description, such as:
    - "Find me red summer dresses"
    - "Show me blue running shoes"
    - "I want casual shirts for men"
    
    The tool returns product information including name, category, color, and other attributes.
    """
    args_schema: type[BaseModel] = TextSearchInput

    embedding_service: EmbeddingService = Field(default=None, exclude=True)
    milvus_service: MilvusService = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        if self.embedding_service is None:
            self.embedding_service = EmbeddingService()
            self.embedding_service.connect_clip()
        if self.milvus_service is None:
            self.milvus_service = MilvusService()
            self.milvus_service.connect()

    def _run(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[str] = None,
    ) -> str:
        """Execute text-based product search

        Args:
            query: Natural language product description
            limit: Maximum number of results
            filters: Optional filter expression

        Returns:
            Formatted string with product information
        """
        try:
            logger.info(f"Searching products with query: '{query}', limit: {limit}")

            # Generate text embedding for query
            query_embedding = self.embedding_service.get_text_embedding(query)

            # Search in Milvus
            results = self.milvus_service.search_similar_text(
                query_embedding=query_embedding,
                limit=limit,
                filters=filters,
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
                return "No products found matching your search criteria."

            # Format results
            formatted_results = self._format_results(results)
            logger.info(f"Found {len(results)} products")

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching products: {e}", exc_info=True)
            return f"Error searching products: {str(e)}"

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into readable text

        Args:
            results: List of product dictionaries from Milvus

        Returns:
            Formatted string representation
        """
        if not results:
            return "No products found."

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

            # Similarity score if available
            if "distance" in product:
                similarity = 1 - product["distance"]  # Convert distance to similarity
                output += f"   Relevance: {similarity:.2%}\n"

            output += "\n"

        return output.strip()


class FilterProductsInput(BaseModel):
    """Input schema for filtering products by attributes"""

    gender: Optional[str] = Field(
        default=None, description="Filter by gender: Men, Women, Boys, Girls, Unisex"
    )
    category: Optional[str] = Field(
        default=None,
        description="Filter by master category: Apparel, Accessories, Footwear",
    )
    color: Optional[str] = Field(
        default=None, description="Filter by color, e.g., Red, Blue, Black"
    )
    season: Optional[str] = Field(
        default=None, description="Filter by season: Summer, Winter, Fall, Spring"
    )
    article_type: Optional[str] = Field(
        default=None, description="Filter by article type, e.g., Shirts, Shoes, Jeans"
    )
    limit: int = Field(
        default=10, description="Maximum number of products to return", ge=1, le=50
    )


class FilterProductsTool(BaseTool):
    """Tool for filtering products by specific attributes"""

    name: str = "filter_products"
    description: str = """
    Filter products by specific attributes like gender, category, color, season, or article type.
    
    Use this tool when users want to browse specific categories or filter by attributes:
    - "Show me all women's shoes"
    - "What red dresses do you have?"
    - "Find men's winter jackets"
    
    This is more precise than search when users specify exact attributes.
    """
    args_schema: type[BaseModel] = FilterProductsInput

    milvus_service: MilvusService = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        if self.milvus_service is None:
            self.milvus_service = MilvusService()
            self.milvus_service.connect()

    def _run(
        self,
        gender: Optional[str] = None,
        category: Optional[str] = None,
        color: Optional[str] = None,
        season: Optional[str] = None,
        article_type: Optional[str] = None,
        limit: int = 10,
    ) -> str:
        """Execute product filtering by attributes

        Args:
            gender: Gender filter
            category: Category filter
            color: Color filter
            season: Season filter
            article_type: Article type filter
            limit: Maximum results

        Returns:
            Formatted string with filtered products
        """
        try:
            # Build filter expression
            filter_parts = []
            if gender:
                filter_parts.append(f'gender == "{gender}"')
            if category:
                filter_parts.append(f'masterCategory == "{category}"')
            if color:
                filter_parts.append(f'baseColour == "{color}"')
            if season:
                filter_parts.append(f'season == "{season}"')
            if article_type:
                filter_parts.append(f'articleType == "{article_type}"')

            filter_expr = " && ".join(filter_parts) if filter_parts else None

            logger.info(f"Filtering products with: {filter_expr}, limit: {limit}")

            # Query Milvus with filters
            # Use a dummy vector search with filters (get any products matching filters)
            dummy_vector = [0.0] * settings.text_dim
            results = self.milvus_service.search_similar_text(
                query_embedding=dummy_vector,
                limit=limit,
                filters=filter_expr,
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
                return "No products found matching your filter criteria."

            # Format results (similar to search)
            output = f"Found {len(results)} product(s) matching filters:\n\n"

            for idx, product in enumerate(results, 1):
                output += (
                    f"{idx}. {product.get('productDisplayName', 'Unknown Product')}\n"
                )
                output += f"   ID: {product.get('id', 'N/A')}\n"
                output += f"   Category: {product.get('masterCategory', 'N/A')} > {product.get('subCategory', 'N/A')} > {product.get('articleType', 'N/A')}\n"
                output += f"   Color: {product.get('baseColour', 'N/A')}\n"
                output += f"   Gender: {product.get('gender', 'N/A')}\n"

                if product.get("season"):
                    output += f"   Season: {product.get('season')}\n"
                if product.get("usage"):
                    output += f"   Usage: {product.get('usage')}\n"

                output += "\n"

            logger.info(f"Found {len(results)} filtered products")
            return output.strip()

        except Exception as e:
            logger.error(f"Error filtering products: {e}", exc_info=True)
            return f"Error filtering products: {str(e)}"


class ImageSearchInput(BaseModel):
    """Input schema for image-based product search"""

    image_path: str = Field(
        description="Path to the image file to search for similar products, e.g., 'data/images/12345.jpg'"
    )
    limit: int = Field(
        default=5,
        description="Maximum number of similar products to return (1-20)",
        ge=1,
        le=20,
    )
    filters: Optional[str] = Field(
        default=None,
        description="Optional filter expression, e.g., 'gender == \"Women\"' or 'masterCategory == \"Apparel\"'",
    )


class ImageSearchTool(BaseTool):
    """Tool for finding similar products using image search"""

    name: str = "search_by_image"
    description: str = """
    Find similar fashion products using an image.
    
    Use this tool when users:
    - Want to find products similar to a specific product image
    - Ask "find similar items to product X"
    - Provide a product ID and want similar recommendations
    
    The tool uses visual similarity (CLIP embeddings) to find products that look similar.
    Input should be a path to an image file, typically in format: 'data/images/{product_id}.jpg'
    """
    args_schema: type[BaseModel] = ImageSearchInput

    embedding_service: EmbeddingService = Field(default=None, exclude=True)
    milvus_service: MilvusService = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        if self.embedding_service is None:
            self.embedding_service = EmbeddingService()
            self.embedding_service.connect_clip()
        if self.milvus_service is None:
            self.milvus_service = MilvusService()
            self.milvus_service.connect()

    def _run(
        self,
        image_path: str,
        limit: int = 5,
        filters: Optional[str] = None,
    ) -> str:
        """Execute image-based product search

        Args:
            image_path: Path to the image file
            limit: Maximum number of results
            filters: Optional filter expression

        Returns:
            Formatted string with similar products
        """
        try:
            logger.info(
                f"Searching similar products for image: '{image_path}', limit: {limit}"
            )

            # Validate image path
            img_path = Path(image_path)
            if not img_path.exists():
                return f"Error: Image file not found at '{image_path}'"

            # Generate image embedding
            image_embedding = self.embedding_service.get_image_embedding(image_path)

            if image_embedding is None:
                return f"Error: Failed to generate embedding for image '{image_path}'"

            # Search in Milvus
            results = self.milvus_service.search_similar_images(
                query_embedding=image_embedding,
                limit=limit + 1,  # +1 to potentially exclude the query image itself
                filters=filters,
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

            # Filter out the query image itself if present
            query_id = img_path.stem  # Extract product ID from filename
            filtered_results = []
            for result in results:
                # Check if this is not the query image
                result_path = result.get("image_path", "")
                if Path(result_path).stem != query_id:
                    filtered_results.append(result)
                    if len(filtered_results) >= limit:
                        break

            if not filtered_results:
                return "No similar products found (excluding the query image)."

            # Format results
            formatted_results = self._format_results(filtered_results, image_path)
            logger.info(f"Found {len(filtered_results)} similar products")

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching by image: {e}", exc_info=True)
            return f"Error searching by image: {str(e)}"

    def _format_results(
        self, results: List[Dict[str, Any]], query_image_path: str
    ) -> str:
        """Format image search results into readable text

        Args:
            results: List of product dictionaries from Milvus
            query_image_path: Path to the query image

        Returns:
            Formatted string representation
        """
        if not results:
            return "No similar products found."

        output = f"Found {len(results)} similar product(s) to '{query_image_path}':\n\n"

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

            # Visual similarity score
            if "distance" in product:
                similarity = 1 - product["distance"]  # Convert distance to similarity
                output += f"   Visual Similarity: {similarity:.2%}\n"

            output += "\n"

        return output.strip()


class VLMAnalysisInput(BaseModel):
    """Input schema for VLM image analysis"""

    image_path: str = Field(
        description="Path to the image file to analyze, e.g., 'data/images/12345.jpg'"
    )
    focus: Optional[str] = Field(
        default=None,
        description="Optional focus area for analysis, e.g., 'color', 'style', 'pattern', 'occasion'",
    )


class VLMReasoningTool(BaseTool):
    """Tool for analyzing fashion product images using Vision Language Model"""

    name: str = "analyze_image_style"
    description: str = """
    Analyze a fashion product image using AI vision to extract detailed style information.
    
    Use this tool when you need to:
    - Understand the style, color, pattern, or design of a product from an image
    - Extract attributes like "casual", "formal", "vintage", "modern", etc.
    - Identify material appearance (e.g., "cotton", "denim", "leather")
    - Determine occasion suitability (e.g., "office wear", "party", "sports")
    - Get detailed descriptions for subsequent text-based searches
    
    This tool uses GPT-4o-mini with vision to provide detailed fashion analysis.
    Input should be a path to an image file, typically: 'data/images/{product_id}.jpg'
    
    Example use case: User uploads a dress image and asks "find similar cocktail dresses"
    -> First use this tool to analyze the dress style
    -> Then use the analysis result for text-based search
    """
    args_schema: type[BaseModel] = VLMAnalysisInput

    openai_client: OpenAI = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        if self.openai_client is None:
            self.openai_client = OpenAI(api_key=settings.openai_api_key)

    def _run(
        self,
        image_path: str,
        focus: Optional[str] = None,
    ) -> str:
        """Execute VLM analysis on fashion product image

        Args:
            image_path: Path to the image file
            focus: Optional focus area for analysis

        Returns:
            Detailed text description of the product's visual attributes
        """
        try:
            logger.info(f"Analyzing image with VLM: '{image_path}'")

            # Validate image path
            img_path = Path(image_path)
            if not img_path.exists():
                return f"Error: Image file not found at '{image_path}'"

            # Read and encode image
            with open(img_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

            # Construct prompt based on focus
            if focus:
                prompt = f"""Analyze this fashion product image with a focus on {focus}.

Provide a detailed description including:
- Main product type (e.g., shirt, dress, shoes, bag)
- Specific focus on: {focus}
- Style characteristics relevant to {focus}
- Any other notable features related to {focus}

Keep the description concise but informative (2-3 sentences)."""
            else:
                prompt = """Analyze this fashion product image and provide a detailed description.

Include the following information:
- Product type (e.g., shirt, dress, shoes, pants, bag, accessory)
- Primary colors
- Style/design (e.g., casual, formal, sporty, vintage, modern)
- Pattern or texture (e.g., plain, striped, checked, floral, solid)
- Key features (e.g., collar type, sleeve length, fit, embellishments)
- Material appearance (if obvious, e.g., denim, cotton, leather)
- Suitable occasion or use case (e.g., office wear, party, casual outing, sports)

Provide a comprehensive yet concise description (3-4 sentences)."""

            # Call GPT-4o-mini with vision
            response = self.openai_client.chat.completions.create(
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
                temperature=0.3,  # Lower temperature for more consistent analysis
            )

            analysis = response.choices[0].message.content.strip()
            logger.info(f"VLM analysis completed for '{image_path}'")

            return analysis

        except FileNotFoundError:
            error_msg = f"Error: Image file not found at '{image_path}'"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            logger.error(f"Error in VLM analysis: {e}", exc_info=True)
            return f"Error analyzing image: {str(e)}"
