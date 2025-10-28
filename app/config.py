"""
Configuration management for OmniShopAgent
Loads environment variables and provides configuration objects
"""

import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables
    
    All settings can be configured via .env file or environment variables.
    """

    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 1000

    # CLIP Server Configuration
    clip_server_url: str = "grpc://localhost:51000"

    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    session_ttl: int = 3600

    # Milvus Configuration
    milvus_uri: str = "./data/milvus_lite.db"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    text_collection_name: str = "text_embeddings"
    image_collection_name: str = "image_embeddings"
    text_dim: int = 1536
    image_dim: int = 512

    @property
    def milvus_uri_absolute(self) -> str:
        """Get absolute path for Milvus URI

        Returns:
            - For http/https URIs: returns as-is (Milvus Standalone)
            - For file paths starting with ./: converts to absolute path (Milvus Lite)
            - For other paths: returns as-is
        """
        import os

        # If it's a network URI, return as-is (Milvus Standalone)
        if self.milvus_uri.startswith(("http://", "https://")):
            return self.milvus_uri
        # If it's a relative path, convert to absolute (Milvus Lite)
        if self.milvus_uri.startswith("./"):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(base_dir, self.milvus_uri[2:])
        # Otherwise return as-is
        return self.milvus_uri

    # Search Configuration
    top_k_results: int = 10
    similarity_threshold: float = 0.6

    # Application Configuration
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True
    log_level: str = "INFO"

    # Data Paths
    raw_data_path: str = "./data/raw"
    processed_data_path: str = "./data/processed"
    image_data_path: str = "./data/images"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Helper function to get absolute paths
def get_absolute_path(relative_path: str) -> str:
    """Convert relative path to absolute path"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, relative_path)
