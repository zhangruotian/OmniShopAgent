"""
Embedding Service for Text and Image Embeddings
Supports OpenAI text embeddings and CLIP image embeddings
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from clip_client import Client as ClipClient
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text and image embeddings"""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        clip_server_url: Optional[str] = None,
    ):
        """Initialize embedding service

        Args:
            openai_api_key: OpenAI API key. If None, uses settings.openai_api_key
            clip_server_url: CLIP server URL. If None, uses settings.clip_server_url
        """
        # Initialize OpenAI client for text embeddings
        self.openai_api_key = openai_api_key or settings.openai_api_key
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.text_embedding_model = settings.openai_embedding_model

        # Initialize CLIP client for image embeddings
        self.clip_server_url = clip_server_url or settings.clip_server_url
        self.clip_client: Optional[ClipClient] = None

        logger.info("Embedding service initialized")

    def connect_clip(self) -> None:
        """Connect to CLIP server"""
        try:
            self.clip_client = ClipClient(server=self.clip_server_url)
            logger.info(f"Connected to CLIP server at {self.clip_server_url}")
        except Exception as e:
            logger.error(f"Failed to connect to CLIP server: {e}")
            raise

    def disconnect_clip(self) -> None:
        """Disconnect from CLIP server"""
        if self.clip_client:
            # Note: clip_client doesn't have explicit close method
            self.clip_client = None
            logger.info("Disconnected from CLIP server")

    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats
        """
        try:
            response = self.openai_client.embeddings.create(
                input=text, model=self.text_embedding_model
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated text embedding for: {text[:50]}...")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {e}")
            raise

    def get_text_embeddings_batch(
        self, texts: List[str], batch_size: int = 100
    ) -> List[List[float]]:
        """Get embeddings for multiple texts in batches

        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            try:
                response = self.openai_client.embeddings.create(
                    input=batch, model=self.text_embedding_model
                )

                # Extract embeddings in the correct order
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)

                logger.info(
                    f"Generated text embeddings for batch {i // batch_size + 1}: {len(embeddings)} embeddings"
                )

            except Exception as e:
                logger.error(
                    f"Failed to generate text embeddings for batch {i // batch_size + 1}: {e}"
                )
                raise

        return all_embeddings

    def get_image_embedding(self, image_path: Union[str, Path]) -> List[float]:
        """Get CLIP embedding for a single image

        Args:
            image_path: Path to image file

        Returns:
            Embedding vector as list of floats
        """
        if not self.clip_client:
            raise RuntimeError("CLIP client not connected. Call connect_clip() first.")

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            # Get embedding from CLIP server using image path (as string)
            result = self.clip_client.encode([str(image_path)])

            # Extract embedding - result is numpy array
            import numpy as np

            if isinstance(result, np.ndarray):
                # If result is numpy array, use first element
                embedding = (
                    result[0].tolist() if len(result.shape) > 1 else result.tolist()
                )
            else:
                # If result is DocumentArray
                embedding = result[0].embedding.tolist()

            logger.debug(f"Generated image embedding for: {image_path.name}")
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate image embedding for {image_path}: {e}")
            raise

    def get_image_embeddings_batch(
        self, image_paths: List[Union[str, Path]], batch_size: int = 32
    ) -> List[Optional[List[float]]]:
        """Get CLIP embeddings for multiple images in batches

        Args:
            image_paths: List of paths to image files
            batch_size: Number of images to process at once

        Returns:
            List of embedding vectors (None for failed images)
        """
        if not self.clip_client:
            raise RuntimeError("CLIP client not connected. Call connect_clip() first.")

        all_embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            valid_paths = []
            valid_indices = []

            # Check which images exist
            for idx, path in enumerate(batch_paths):
                path = Path(path)
                if path.exists():
                    valid_paths.append(str(path))
                    valid_indices.append(idx)
                else:
                    logger.warning(f"Image not found: {path}")

            # Get embeddings for valid images
            if valid_paths:
                try:
                    # Send paths as strings to CLIP server
                    result = self.clip_client.encode(valid_paths)

                    # Create embeddings list with None for missing images
                    batch_embeddings = [None] * len(batch_paths)

                    # Handle result format - could be numpy array or DocumentArray
                    import numpy as np

                    if isinstance(result, np.ndarray):
                        # Result is numpy array - shape (n_images, embedding_dim)
                        for idx in range(len(result)):
                            original_idx = valid_indices[idx]
                            batch_embeddings[original_idx] = result[idx].tolist()
                    else:
                        # Result is DocumentArray
                        for idx, doc in enumerate(result):
                            original_idx = valid_indices[idx]
                            batch_embeddings[original_idx] = doc.embedding.tolist()

                    all_embeddings.extend(batch_embeddings)

                    logger.info(
                        f"Generated image embeddings for batch {i // batch_size + 1}: "
                        f"{len(valid_paths)}/{len(batch_paths)} successful"
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to generate image embeddings for batch {i // batch_size + 1}: {e}"
                    )
                    # Add None for all images in failed batch
                    all_embeddings.extend([None] * len(batch_paths))
            else:
                # All images in batch failed to load
                all_embeddings.extend([None] * len(batch_paths))

        return all_embeddings

    def get_text_embedding_from_image(
        self, image_path: Union[str, Path]
    ) -> List[float]:
        """Get text-based embedding by describing the image
        This is useful for cross-modal search

        Note: This is a placeholder for future implementation
        that could use vision models to generate text descriptions

        Args:
            image_path: Path to image file

        Returns:
            Text embedding vector
        """
        # For now, we just return the image embedding
        # In the future, this could use a vision-language model to generate
        # a text description and then embed that
        raise NotImplementedError("Text embedding from image not yet implemented")

    def cosine_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Normalize vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)

        # Calculate cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)

        return float(similarity)

    def get_embedding_dimensions(self) -> dict:
        """Get the dimensions of text and image embeddings

        Returns:
            Dictionary with text_dim and image_dim
        """
        return {"text_dim": settings.text_dim, "image_dim": settings.image_dim}


# Global instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
        _embedding_service.connect_clip()
    return _embedding_service
