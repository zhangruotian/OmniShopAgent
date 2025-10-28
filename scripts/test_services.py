"""
Test Script for Services
Verifies that all services are working correctly
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_absolute_path, settings
from app.services.embedding_service import EmbeddingService
from app.services.milvus_service import MilvusService
from app.services.mongodb_service import MongoDBService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_mongodb():
    """Test MongoDB service"""
    print("\n" + "=" * 60)
    print("Testing MongoDB Service")
    print("=" * 60)

    try:
        service = MongoDBService()
        service.connect()

        # Test connection
        print("✓ MongoDB connection successful")

        # Get statistics
        stats = service.get_stats()
        print(f"✓ Total products: {stats['total_products']}")
        print(f"  Gender distribution: {stats['gender_distribution']}")
        print(f"  Master categories: {list(stats['master_categories'].keys())}")

        # Test query
        if stats["total_products"] > 0:
            products = service.search_products(filters={"gender": "Men"}, limit=3)
            print(f"✓ Query test: Found {len(products)} men's products")
            if products:
                print(f"  Example: {products[0]['productDisplayName']}")

        service.disconnect()
        print("✓ MongoDB service test passed!")
        return True

    except Exception as e:
        print(f"✗ MongoDB service test failed: {e}")
        return False


def test_embedding_service():
    """Test Embedding service"""
    print("\n" + "=" * 60)
    print("Testing Embedding Service")
    print("=" * 60)

    try:
        service = EmbeddingService()

        # Test OpenAI text embedding
        print("Testing OpenAI text embedding...")
        text = "Blue denim jeans for men"
        embedding = service.get_text_embedding(text)
        print(f"✓ Text embedding generated: {len(embedding)} dimensions")

        # Test CLIP image embedding
        print("Testing CLIP image embedding...")
        service.connect_clip()
        print("✓ Connected to CLIP server")

        # Find a test image
        image_dir = Path(get_absolute_path(settings.image_data_path))
        test_images = list(image_dir.glob("*.jpg"))[:1]

        if test_images:
            image_path = test_images[0]
            print(f"  Using test image: {image_path.name}")
            image_embedding = service.get_image_embedding(image_path)
            print(f"✓ Image embedding generated: {len(image_embedding)} dimensions")
        else:
            print("! No test images found, skipping image embedding test")

        # Test batch processing
        texts = ["Men's shirt", "Women's dress", "Kids shoes"]
        embeddings = service.get_text_embeddings_batch(texts, batch_size=3)
        print(f"✓ Batch text embeddings: {len(embeddings)} embeddings generated")

        service.disconnect_clip()
        print("✓ Embedding service test passed!")
        return True

    except Exception as e:
        print(f"✗ Embedding service test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_milvus_service():
    """Test Milvus service"""
    print("\n" + "=" * 60)
    print("Testing Milvus Service")
    print("=" * 60)

    try:
        service = MilvusService()
        service.connect()
        print("✓ Milvus connection successful")

        # Create collections (if not exist)
        service.create_text_collection(recreate=False)
        print("✓ Text collection ready")

        service.create_image_collection(recreate=False)
        print("✓ Image collection ready")

        # Get statistics
        text_stats = service.get_collection_stats(service.text_collection_name)
        image_stats = service.get_collection_stats(service.image_collection_name)
        print(f"✓ Text embeddings count: {text_stats['row_count']}")
        print(f"✓ Image embeddings count: {image_stats['row_count']}")

        # Test search (if data exists)
        if text_stats["row_count"] > 0:
            # Get a sample embedding
            import numpy as np

            query_embedding = np.random.rand(settings.text_dim).tolist()

            results = service.search_similar_text(
                query_embedding=query_embedding, limit=3
            )
            print(f"✓ Text search test: Found {len(results)} similar items")

        if image_stats["row_count"] > 0:
            # Get a sample embedding
            import numpy as np

            query_embedding = np.random.rand(settings.image_dim).tolist()

            results = service.search_similar_images(
                query_embedding=query_embedding, limit=3
            )
            print(f"✓ Image search test: Found {len(results)} similar items")

        service.disconnect()
        print("✓ Milvus service test passed!")
        return True

    except Exception as e:
        print(f"✗ Milvus service test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_integration():
    """Test integration between services"""
    print("\n" + "=" * 60)
    print("Testing Service Integration")
    print("=" * 60)

    try:
        # Initialize all services
        mongo = MongoDBService()
        embed = EmbeddingService()
        milvus = MilvusService()

        mongo.connect()
        embed.connect_clip()
        milvus.connect()

        print("✓ All services connected")

        # Get a product from MongoDB
        products = mongo.get_products_batch(batch_size=1)
        if not products:
            print("! No products in MongoDB, skipping integration test")
            return True

        product = products[0]
        print(f"✓ Retrieved product: {product['productDisplayName']}")

        # Generate text embedding
        text = f"{product['productDisplayName']} {product['articleType']} {product['baseColour']}"
        text_embedding = embed.get_text_embedding(text)
        print(f"✓ Generated text embedding: {len(text_embedding)} dimensions")

        # Search for similar products in Milvus
        text_stats = milvus.get_collection_stats(milvus.text_collection_name)
        if text_stats["row_count"] > 0:
            results = milvus.search_similar_text(
                query_embedding=text_embedding, limit=5
            )
            print(f"✓ Found {len(results)} similar products")

            # Retrieve full product details from MongoDB
            if results:
                product_ids = [r["product_id"] for r in results[:3]]
                for pid in product_ids:
                    p = mongo.get_product_by_id(pid)
                    if p:
                        print(
                            f"  - {p['productDisplayName']} (distance: {results[product_ids.index(pid)]['distance']:.4f})"
                        )

        # Clean up
        mongo.disconnect()
        embed.disconnect_clip()
        milvus.disconnect()

        print("✓ Integration test passed!")
        return True

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("OmniShopAgent Service Tests")
    print("=" * 60)

    results = {}

    # Run individual service tests
    results["mongodb"] = test_mongodb()
    results["embedding"] = test_embedding_service()
    results["milvus"] = test_milvus_service()

    # Run integration test
    results["integration"] = test_integration()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.capitalize()}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
