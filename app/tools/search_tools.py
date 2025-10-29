"""
Search Tools for Product Discovery
Provides text-based and image-based product search capabilities
"""

import logging
from typing import Any, Dict, List, Optional

from langchain.tools import BaseTool
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

    async def _arun(self, *args, **kwargs) -> str:
        """Async version (not implemented, falls back to sync)"""
        return self._run(*args, **kwargs)

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
                output += f"{idx}. {product.get('productDisplayName', 'Unknown Product')}\n"
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

    async def _arun(self, *args, **kwargs) -> str:
        """Async version (not implemented, falls back to sync)"""
        return self._run(*args, **kwargs)

