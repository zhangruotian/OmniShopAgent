"""
Safe script to view Milvus statistics without recreating collections
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct import to avoid circular dependencies
import importlib.util

from app.config import settings

spec = importlib.util.spec_from_file_location(
    "milvus_service",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "app/services/milvus_service.py",
    ),
)
milvus_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(milvus_module)
MilvusService = milvus_module.MilvusService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """View Milvus statistics safely"""
    milvus = MilvusService()

    try:
        # Connect to Milvus
        milvus.connect()
        logger.info("Connected to Milvus")

        # Get collection statistics
        text_stats = milvus.get_collection_stats(settings.text_collection_name)
        image_stats = milvus.get_collection_stats(settings.image_collection_name)

        print("\n" + "=" * 80)
        print("MILVUS DATA STATISTICS".center(80))
        print("=" * 80)

        print(f"\n📝 {settings.text_collection_name}")
        print(f"   Collection: {text_stats.get('collection_name', 'N/A')}")
        print(f"   Entity Count: {text_stats.get('row_count', 0):,}")
        print(f"   Dimension: {settings.text_dim}")

        print(f"\n🖼️  {settings.image_collection_name}")
        print(f"   Collection: {image_stats.get('collection_name', 'N/A')}")
        print(f"   Entity Count: {image_stats.get('row_count', 0):,}")
        print(f"   Dimension: {settings.image_dim}")

        print("\n📊 Expected Count: 44,446")

        text_count = text_stats.get("row_count", 0)
        image_count = image_stats.get("row_count", 0)

        if text_count == 44446 and image_count == 44446:
            print("✅ Data count is CORRECT!")
        else:
            print("⚠️  Data count mismatch:")
            if text_count != 44446:
                print(
                    f"   Text: Expected 44,446, got {text_count:,} (diff: {text_count - 44446:+,})"
                )
            if image_count != 44446:
                print(
                    f"   Image: Expected 44,446, got {image_count:,} (diff: {image_count - 44446:+,})"
                )

        print("\n" + "=" * 80 + "\n")

    finally:
        milvus.disconnect()
        logger.info("Disconnected from Milvus")


if __name__ == "__main__":
    main()
