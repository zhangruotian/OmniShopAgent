"""
Data Indexing Script
Generates embeddings for products and stores them in Milvus
"""

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import config and settings first
# Direct imports from files to avoid __init__.py circular issues
import importlib.util

from app.config import get_absolute_path, settings


def load_service_module(module_name, file_name):
    """Load a service module directly from file"""
    spec = importlib.util.spec_from_file_location(
        module_name,
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            f"app/services/{file_name}",
        ),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


embedding_module = load_service_module("embedding_service", "embedding_service.py")
milvus_module = load_service_module("milvus_service", "milvus_service.py")

EmbeddingService = embedding_module.EmbeddingService
MilvusService = milvus_module.MilvusService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataIndexer:
    """Index product data by generating and storing embeddings"""

    def __init__(self):
        """Initialize services"""
        self.embedding_service = EmbeddingService()
        self.milvus_service = MilvusService()

        self.image_dir = Path(get_absolute_path(settings.image_data_path))
        self.styles_csv = get_absolute_path("./data/styles.csv")
        self.images_csv = get_absolute_path("./data/images.csv")

        # Load product data from CSV
        self.products = self._load_products_from_csv()

    def _load_products_from_csv(self) -> Dict[int, Dict[str, Any]]:
        """Load products from CSV files"""
        products = {}

        # Load images mapping
        images_dict = {}
        with open(self.images_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                product_id = int(row["filename"].split(".")[0])
                images_dict[product_id] = row["link"]

        # Load styles/products
        with open(self.styles_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    product_id = int(row["id"])
                    products[product_id] = {
                        "id": product_id,
                        "gender": row.get("gender", ""),
                        "masterCategory": row.get("masterCategory", ""),
                        "subCategory": row.get("subCategory", ""),
                        "articleType": row.get("articleType", ""),
                        "baseColour": row.get("baseColour", ""),
                        "season": row.get("season", ""),
                        "year": int(row["year"]) if row.get("year") else 0,
                        "usage": row.get("usage", ""),
                        "productDisplayName": row.get("productDisplayName", ""),
                        "imageUrl": images_dict.get(product_id, ""),
                        "imagePath": f"{product_id}.jpg",
                    }
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error loading product {row.get('id')}: {e}")
                    continue

        logger.info(f"Loaded {len(products)} products from CSV")
        return products

    def setup(self) -> None:
        """Setup connections and collections"""
        logger.info("Setting up services...")

        # Connect to CLIP server
        self.embedding_service.connect_clip()
        logger.info("✓ CLIP server connected")

        # Connect to Milvus
        self.milvus_service.connect()
        logger.info("✓ Milvus connected")

        # Create Milvus collections
        self.milvus_service.create_text_collection(recreate=False)
        self.milvus_service.create_image_collection(recreate=False)
        logger.info("✓ Milvus collections ready")

    def teardown(self) -> None:
        """Close all connections"""
        logger.info("Closing connections...")
        self.embedding_service.disconnect_clip()
        self.milvus_service.disconnect()
        logger.info("✓ All connections closed")

    def index_text_embeddings(
        self, batch_size: int = 100, skip: int = 0, limit: Optional[int] = None
    ) -> Dict[str, int]:
        """Generate and store text embeddings for products

        Args:
            batch_size: Number of products to process at once
            skip: Number of products to skip
            limit: Maximum number of products to process (None for all)

        Returns:
            Dictionary with indexing statistics
        """
        logger.info("Starting text embedding indexing...")

        # Get products list
        product_ids = list(self.products.keys())[skip:]
        if limit:
            product_ids = product_ids[:limit]

        total_products = len(product_ids)
        processed = 0
        inserted = 0
        errors = 0

        with tqdm(total=total_products, desc="Indexing text embeddings") as pbar:
            while processed < total_products:
                # Get batch of products
                current_batch_size = min(batch_size, total_products - processed)
                batch_ids = product_ids[processed : processed + current_batch_size]
                products = [self.products[pid] for pid in batch_ids]

                if not products:
                    break

                try:
                    # Prepare texts for embedding
                    texts = []
                    text_mappings = []

                    for product in products:
                        # Create text representation of product
                        text = self._create_product_text(product)
                        texts.append(text)
                        text_mappings.append(
                            {"product_id": product["id"], "text": text}
                        )

                    # Generate embeddings
                    embeddings = self.embedding_service.get_text_embeddings_batch(
                        texts, batch_size=50  # OpenAI batch size
                    )

                    # Prepare data for Milvus (with metadata)
                    milvus_data = []
                    for idx, (mapping, embedding) in enumerate(
                        zip(text_mappings, embeddings)
                    ):
                        product_id = mapping["product_id"]
                        product = self.products[product_id]

                        milvus_data.append(
                            {
                                "id": product_id,
                                "text": mapping["text"][
                                    :2000
                                ],  # Truncate to max length
                                "embedding": embedding,
                                # Product metadata
                                "productDisplayName": product["productDisplayName"][
                                    :500
                                ],
                                "gender": product["gender"][:50],
                                "masterCategory": product["masterCategory"][:100],
                                "subCategory": product["subCategory"][:100],
                                "articleType": product["articleType"][:100],
                                "baseColour": product["baseColour"][:50],
                                "season": product["season"][:50],
                                "usage": product["usage"][:50],
                                "year": product["year"],
                                "imageUrl": product["imageUrl"],
                                "imagePath": product["imagePath"],
                            }
                        )

                    # Insert into Milvus
                    count = self.milvus_service.insert_text_embeddings(milvus_data)
                    inserted += count

                except Exception as e:
                    logger.error(
                        f"Error processing text batch at offset {processed}: {e}"
                    )
                    errors += len(products)

                processed += len(products)
                pbar.update(len(products))

        stats = {"total_processed": processed, "inserted": inserted, "errors": errors}

        logger.info(f"Text embedding indexing completed: {stats}")
        return stats

    def index_image_embeddings(
        self, batch_size: int = 32, skip: int = 0, limit: Optional[int] = None
    ) -> Dict[str, int]:
        """Generate and store image embeddings for products

        Args:
            batch_size: Number of images to process at once
            skip: Number of products to skip
            limit: Maximum number of products to process (None for all)

        Returns:
            Dictionary with indexing statistics
        """
        logger.info("Starting image embedding indexing...")

        # Get products list
        product_ids = list(self.products.keys())[skip:]
        if limit:
            product_ids = product_ids[:limit]

        total_products = len(product_ids)
        processed = 0
        inserted = 0
        errors = 0

        with tqdm(total=total_products, desc="Indexing image embeddings") as pbar:
            while processed < total_products:
                # Get batch of products
                current_batch_size = min(batch_size, total_products - processed)
                batch_ids = product_ids[processed : processed + current_batch_size]
                products = [self.products[pid] for pid in batch_ids]

                if not products:
                    break

                try:
                    # Prepare image paths
                    image_paths = []
                    image_mappings = []

                    for product in products:
                        image_path = self.image_dir / product["imagePath"]
                        image_paths.append(image_path)
                        image_mappings.append(
                            {
                                "product_id": product["id"],
                                "image_path": product["imagePath"],
                            }
                        )

                    # Generate embeddings
                    embeddings = self.embedding_service.get_image_embeddings_batch(
                        image_paths, batch_size=batch_size
                    )

                    # Prepare data for Milvus (with metadata)
                    milvus_data = []
                    for idx, (mapping, embedding) in enumerate(
                        zip(image_mappings, embeddings)
                    ):
                        if embedding is not None:
                            product_id = mapping["product_id"]
                            product = self.products[product_id]

                            milvus_data.append(
                                {
                                    "id": product_id,
                                    "image_path": mapping["image_path"],
                                    "embedding": embedding,
                                    # Product metadata
                                    "productDisplayName": product["productDisplayName"][
                                        :500
                                    ],
                                    "gender": product["gender"][:50],
                                    "masterCategory": product["masterCategory"][:100],
                                    "subCategory": product["subCategory"][:100],
                                    "articleType": product["articleType"][:100],
                                    "baseColour": product["baseColour"][:50],
                                    "season": product["season"][:50],
                                    "usage": product["usage"][:50],
                                    "year": product["year"],
                                    "imageUrl": product["imageUrl"],
                                }
                            )
                        else:
                            errors += 1

                    # Insert into Milvus
                    if milvus_data:
                        count = self.milvus_service.insert_image_embeddings(milvus_data)
                        inserted += count

                except Exception as e:
                    logger.error(
                        f"Error processing image batch at offset {processed}: {e}"
                    )
                    errors += len(products)

                processed += len(products)
                pbar.update(len(products))

        stats = {"total_processed": processed, "inserted": inserted, "errors": errors}

        logger.info(f"Image embedding indexing completed: {stats}")
        return stats

    def _create_product_text(self, product: Dict[str, Any]) -> str:
        """Create text representation of product for embedding

        Args:
            product: Product document

        Returns:
            Text representation
        """
        # Create a natural language description
        parts = [
            product.get("productDisplayName", ""),
            f"Gender: {product.get('gender', '')}",
            f"Category: {product.get('masterCategory', '')} > {product.get('subCategory', '')}",
            f"Type: {product.get('articleType', '')}",
            f"Color: {product.get('baseColour', '')}",
            f"Season: {product.get('season', '')}",
            f"Usage: {product.get('usage', '')}",
        ]

        text = " | ".join(
            [p for p in parts if p and p != "Gender: " and p != "Color: "]
        )
        return text

    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics

        Returns:
            Dictionary with statistics
        """
        text_stats = self.milvus_service.get_collection_stats(
            self.milvus_service.text_collection_name
        )
        image_stats = self.milvus_service.get_collection_stats(
            self.milvus_service.image_collection_name
        )

        return {
            "total_products": len(self.products),
            "milvus_text": text_stats,
            "milvus_image": image_stats,
        }


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Index product data for search")
    parser.add_argument(
        "--mode",
        choices=["text", "image", "both"],
        default="both",
        help="Which embeddings to index",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for processing"
    )
    parser.add_argument(
        "--skip", type=int, default=0, help="Number of products to skip"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of products to process"
    )
    parser.add_argument("--stats", action="store_true", help="Show statistics only")

    args = parser.parse_args()

    # Create indexer
    indexer = DataIndexer()

    try:
        # Setup services
        indexer.setup()

        if args.stats:
            # Show statistics
            stats = indexer.get_stats()
            print("\n=== Indexing Statistics ===")
            print(f"\nTotal Products in CSV: {stats['total_products']}")

            print("\nMilvus Text Embeddings:")
            print(f"  Collection: {stats['milvus_text']['collection_name']}")
            print(f"  Total embeddings: {stats['milvus_text']['row_count']}")

            print("\nMilvus Image Embeddings:")
            print(f"  Collection: {stats['milvus_image']['collection_name']}")
            print(f"  Total embeddings: {stats['milvus_image']['row_count']}")

            print(
                f"\nCoverage: {stats['milvus_image']['row_count'] / stats['total_products'] * 100:.1f}%"
            )
        else:
            # Index data
            if args.mode in ["text", "both"]:
                logger.info("=== Indexing Text Embeddings ===")
                text_stats = indexer.index_text_embeddings(
                    batch_size=args.batch_size, skip=args.skip, limit=args.limit
                )
                print(f"\nText Indexing Results: {text_stats}")

            if args.mode in ["image", "both"]:
                logger.info("=== Indexing Image Embeddings ===")
                image_stats = indexer.index_image_embeddings(
                    batch_size=min(args.batch_size, 32),  # Smaller batch for images
                    skip=args.skip,
                    limit=args.limit,
                )
                print(f"\nImage Indexing Results: {image_stats}")

            # Show final statistics
            logger.info("\n=== Final Statistics ===")
            stats = indexer.get_stats()
            print(f"Total products: {stats['total_products']}")
            print(f"Text embeddings: {stats['milvus_text']['row_count']}")
            print(f"Image embeddings: {stats['milvus_image']['row_count']}")

    except KeyboardInterrupt:
        logger.info("\nIndexing interrupted by user")
    except Exception as e:
        logger.error(f"Error during indexing: {e}", exc_info=True)
        sys.exit(1)
    finally:
        indexer.teardown()


if __name__ == "__main__":
    main()
