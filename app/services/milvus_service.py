"""
Milvus Service for Vector Storage and Similarity Search
Manages text and image embeddings in separate collections
"""

import logging
from typing import Any, Dict, List, Optional

from pymilvus import (
    DataType,
    MilvusClient,
)

from app.config import settings

logger = logging.getLogger(__name__)


class MilvusService:
    """Service for managing vector embeddings in Milvus"""

    def __init__(self, uri: Optional[str] = None):
        """Initialize Milvus service

        Args:
            uri: Milvus connection URI. If None, uses settings.milvus_uri
        """
        if uri:
            self.uri = uri
        else:
            # Use absolute path for Milvus Lite
            self.uri = settings.milvus_uri_absolute
        self.text_collection_name = settings.text_collection_name
        self.image_collection_name = settings.image_collection_name
        self.text_dim = settings.text_dim
        self.image_dim = settings.image_dim

        # Use MilvusClient for simplified operations
        self._client: Optional[MilvusClient] = None

        logger.info(f"Initializing Milvus service with URI: {self.uri}")

    def connect(self) -> None:
        """Connect to Milvus"""
        try:
            self._client = MilvusClient(uri=self.uri)
            logger.info(f"Connected to Milvus at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from Milvus"""
        if self._client:
            self._client.close()
            self._client = None
            logger.info("Disconnected from Milvus")

    @property
    def client(self) -> MilvusClient:
        """Get the Milvus client"""
        if not self._client:
            raise RuntimeError("Milvus not connected. Call connect() first.")
        return self._client

    def create_text_collection(self, recreate: bool = False) -> None:
        """Create collection for text embeddings with product metadata

        Args:
            recreate: If True, drop existing collection and recreate
        """
        if recreate and self.client.has_collection(self.text_collection_name):
            self.client.drop_collection(self.text_collection_name)
            logger.info(f"Dropped existing collection: {self.text_collection_name}")

        if self.client.has_collection(self.text_collection_name):
            logger.info(f"Text collection already exists: {self.text_collection_name}")
            return

        # Create collection with schema (includes metadata fields)
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,  # Allow additional metadata fields
        )

        # Core fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2000)
        schema.add_field(
            field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.text_dim
        )

        # Product metadata fields
        schema.add_field(
            field_name="productDisplayName", datatype=DataType.VARCHAR, max_length=500
        )
        schema.add_field(field_name="gender", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(
            field_name="masterCategory", datatype=DataType.VARCHAR, max_length=100
        )
        schema.add_field(
            field_name="subCategory", datatype=DataType.VARCHAR, max_length=100
        )
        schema.add_field(
            field_name="articleType", datatype=DataType.VARCHAR, max_length=100
        )
        schema.add_field(
            field_name="baseColour", datatype=DataType.VARCHAR, max_length=50
        )
        schema.add_field(field_name="season", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="usage", datatype=DataType.VARCHAR, max_length=50)

        # Create index parameters
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        # Create collection
        self.client.create_collection(
            collection_name=self.text_collection_name,
            schema=schema,
            index_params=index_params,
        )

        logger.info(
            f"Created text collection with metadata: {self.text_collection_name}"
        )

    def create_image_collection(self, recreate: bool = False) -> None:
        """Create collection for image embeddings with product metadata

        Args:
            recreate: If True, drop existing collection and recreate
        """
        if recreate and self.client.has_collection(self.image_collection_name):
            self.client.drop_collection(self.image_collection_name)
            logger.info(f"Dropped existing collection: {self.image_collection_name}")

        if self.client.has_collection(self.image_collection_name):
            logger.info(
                f"Image collection already exists: {self.image_collection_name}"
            )
            return

        # Create collection with schema (includes metadata fields)
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,  # Allow additional metadata fields
        )

        # Core fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="image_path", datatype=DataType.VARCHAR, max_length=500
        )
        schema.add_field(
            field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=self.image_dim
        )

        # Product metadata fields
        schema.add_field(
            field_name="productDisplayName", datatype=DataType.VARCHAR, max_length=500
        )
        schema.add_field(field_name="gender", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(
            field_name="masterCategory", datatype=DataType.VARCHAR, max_length=100
        )
        schema.add_field(
            field_name="subCategory", datatype=DataType.VARCHAR, max_length=100
        )
        schema.add_field(
            field_name="articleType", datatype=DataType.VARCHAR, max_length=100
        )
        schema.add_field(
            field_name="baseColour", datatype=DataType.VARCHAR, max_length=50
        )
        schema.add_field(field_name="season", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="usage", datatype=DataType.VARCHAR, max_length=50)

        # Create index parameters
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        # Create collection
        self.client.create_collection(
            collection_name=self.image_collection_name,
            schema=schema,
            index_params=index_params,
        )

        logger.info(
            f"Created image collection with metadata: {self.image_collection_name}"
        )

    def insert_text_embeddings(
        self,
        embeddings: List[Dict[str, Any]],
    ) -> int:
        """Insert text embeddings with metadata into collection

        Args:
            embeddings: List of dictionaries with keys:
                - id: unique ID (product ID)
                - text: the text that was embedded
                - embedding: the embedding vector
                - productDisplayName, gender, masterCategory, etc. (metadata)

        Returns:
            Number of inserted embeddings
        """
        if not embeddings:
            return 0

        try:
            # Insert data directly (all fields including metadata)
            # Milvus will accept all fields defined in schema + dynamic fields
            data = embeddings

            # Insert data
            result = self.client.insert(
                collection_name=self.text_collection_name,
                data=data,
            )

            logger.info(f"Inserted {len(data)} text embeddings")
            return len(data)

        except Exception as e:
            logger.error(f"Failed to insert text embeddings: {e}")
            raise

    def insert_image_embeddings(
        self,
        embeddings: List[Dict[str, Any]],
    ) -> int:
        """Insert image embeddings with metadata into collection

        Args:
            embeddings: List of dictionaries with keys:
                - id: unique ID (product ID)
                - image_path: path to the image file
                - embedding: the embedding vector
                - productDisplayName, gender, masterCategory, etc. (metadata)

        Returns:
            Number of inserted embeddings
        """
        if not embeddings:
            return 0

        try:
            # Insert data directly (all fields including metadata)
            # Milvus will accept all fields defined in schema + dynamic fields
            data = embeddings

            # Insert data
            result = self.client.insert(
                collection_name=self.image_collection_name,
                data=data,
            )

            logger.info(f"Inserted {len(data)} image embeddings")
            return len(data)

        except Exception as e:
            logger.error(f"Failed to insert image embeddings: {e}")
            raise

    def search_similar_text(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar text embeddings

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filters: Filter expression (e.g., "product_id in [1, 2, 3]")
            output_fields: List of fields to return

        Returns:
            List of search results with fields:
                - id: embedding ID
                - distance: similarity distance
                - entity: the matched entity with requested fields
        """
        try:
            if output_fields is None:
                output_fields = [
                    "id",
                    "text",
                    "productDisplayName",
                    "gender",
                    "masterCategory",
                    "subCategory",
                    "articleType",
                    "baseColour",
                ]

            search_params = {}
            if filters:
                search_params["expr"] = filters

            results = self.client.search(
                collection_name=self.text_collection_name,
                data=[query_embedding],
                limit=limit,
                output_fields=output_fields,
                search_params=search_params,
            )

            # Format results
            formatted_results = []
            if results and len(results) > 0:
                for hit in results[0]:
                    result = {"id": hit.get("id"), "distance": hit.get("distance")}
                    # Extract fields from entity
                    entity = hit.get("entity", {})
                    for field in output_fields:
                        if field in entity:
                            result[field] = entity.get(field)
                    formatted_results.append(result)

            logger.debug(f"Found {len(formatted_results)} similar text embeddings")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search similar text: {e}")
            raise

    def search_similar_images(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar image embeddings

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            filters: Filter expression (e.g., "product_id in [1, 2, 3]")
            output_fields: List of fields to return

        Returns:
            List of search results with fields:
                - id: embedding ID
                - distance: similarity distance
                - entity: the matched entity with requested fields
        """
        try:
            if output_fields is None:
                output_fields = [
                    "id",
                    "image_path",
                    "productDisplayName",
                    "gender",
                    "masterCategory",
                    "subCategory",
                    "articleType",
                    "baseColour",
                ]

            search_params = {}
            if filters:
                search_params["expr"] = filters

            results = self.client.search(
                collection_name=self.image_collection_name,
                data=[query_embedding],
                limit=limit,
                output_fields=output_fields,
                search_params=search_params,
            )

            # Format results
            formatted_results = []
            if results and len(results) > 0:
                for hit in results[0]:
                    result = {"id": hit.get("id"), "distance": hit.get("distance")}
                    # Extract fields from entity
                    entity = hit.get("entity", {})
                    for field in output_fields:
                        if field in entity:
                            result[field] = entity.get(field)
                    formatted_results.append(result)

            logger.debug(f"Found {len(formatted_results)} similar image embeddings")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search similar images: {e}")
            raise

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = self.client.get_collection_stats(collection_name)
            return {
                "collection_name": collection_name,
                "row_count": stats.get("row_count", 0),
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"collection_name": collection_name, "row_count": 0}

    def delete_by_ids(self, collection_name: str, ids: List[int]) -> int:
        """Delete embeddings by IDs

        Args:
            collection_name: Name of the collection
            ids: List of IDs to delete

        Returns:
            Number of deleted embeddings
        """
        if not ids:
            return 0

        try:
            self.client.delete(
                collection_name=collection_name,
                ids=ids,
            )
            logger.info(f"Deleted {len(ids)} embeddings from {collection_name}")
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            raise

    def clear_collection(self, collection_name: str) -> None:
        """Clear all data from a collection

        Args:
            collection_name: Name of the collection
        """
        try:
            if self.client.has_collection(collection_name):
                self.client.drop_collection(collection_name)
                logger.info(f"Dropped collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise


# Global instance
_milvus_service: Optional[MilvusService] = None


def get_milvus_service() -> MilvusService:
    """Get or create the global Milvus service instance"""
    global _milvus_service
    if _milvus_service is None:
        _milvus_service = MilvusService()
        _milvus_service.connect()
    return _milvus_service
