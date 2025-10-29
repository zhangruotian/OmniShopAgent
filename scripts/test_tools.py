"""
Test script for LangChain tools
Tests product search and filter functionality
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tools import (
    FilterProductsTool,
    ImageSearchTool,
    ProductSearchTool,
    VLMReasoningTool,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_product_search():
    """Test text-based product search"""
    print("\n" + "=" * 80)
    print("TEST 1: Product Search Tool".center(80))
    print("=" * 80 + "\n")

    tool = ProductSearchTool()

    # Test cases
    test_queries = [
        {"query": "red summer dress for women", "limit": 3},
        {"query": "blue running shoes", "limit": 3},
        {"query": "casual men's shirt", "limit": 3},
    ]

    for idx, test in enumerate(test_queries, 1):
        print(f"\nTest Case {idx}: '{test['query']}'")
        print("-" * 80)

        try:
            result = tool._run(**test)
            print(result)
            print("\n✅ Test passed")
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            logger.error(f"Error in test case {idx}", exc_info=True)


def test_filter_products():
    """Test product filtering by attributes"""
    print("\n" + "=" * 80)
    print("TEST 2: Filter Products Tool".center(80))
    print("=" * 80 + "\n")

    tool = FilterProductsTool()

    # Test cases
    test_filters = [
        {"gender": "Women", "category": "Apparel", "limit": 5},
        {"color": "Blue", "article_type": "Shoes", "limit": 5},
        {"gender": "Men", "season": "Summer", "limit": 5},
    ]

    for idx, test in enumerate(test_filters, 1):
        filter_desc = ", ".join([f"{k}={v}" for k, v in test.items()])
        print(f"\nTest Case {idx}: Filters({filter_desc})")
        print("-" * 80)

        try:
            result = tool._run(**test)
            print(result)
            print("\n✅ Test passed")
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            logger.error(f"Error in filter test case {idx}", exc_info=True)


def test_image_search():
    """Test image-based product search"""
    print("\n" + "=" * 80)
    print("TEST 3: Image Search Tool".center(80))
    print("=" * 80 + "\n")

    tool = ImageSearchTool()

    # Find some sample images first
    import os
    from pathlib import Path

    image_dir = Path("data/images")
    if not image_dir.exists():
        print("⚠️  Skipping image search test: data/images directory not found")
        return

    # Get a few sample images
    sample_images = list(image_dir.glob("*.jpg"))[:3]

    if not sample_images:
        print("⚠️  Skipping image search test: no images found in data/images")
        return

    for idx, image_path in enumerate(sample_images, 1):
        print(f"\nTest Case {idx}: Find similar products to '{image_path.name}'")
        print("-" * 80)

        try:
            result = tool._run(image_path=str(image_path), limit=3)
            print(result)
            print("\n✅ Test passed")
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            logger.error(f"Error in image search test case {idx}", exc_info=True)


def test_vlm_reasoning():
    """Test VLM image analysis"""
    print("\n" + "=" * 80)
    print("TEST 4: VLM Reasoning Tool".center(80))
    print("=" * 80 + "\n")

    tool = VLMReasoningTool()

    # Find sample images
    from pathlib import Path

    image_dir = Path("data/images")
    if not image_dir.exists():
        print("⚠️  Skipping VLM test: data/images directory not found")
        return

    # Get a few sample images
    sample_images = list(image_dir.glob("*.jpg"))[:2]

    if not sample_images:
        print("⚠️  Skipping VLM test: no images found in data/images")
        return

    for idx, image_path in enumerate(sample_images, 1):
        print(f"\nTest Case {idx}: Analyze style of '{image_path.name}'")
        print("-" * 80)

        try:
            # Test without focus
            result = tool._run(image_path=str(image_path))
            print(f"Analysis:\n{result}")
            print("\n✅ Test passed")

            # Test with focus
            print(f"\nTest Case {idx}b: Analyze with focus on 'color and style'")
            print("-" * 80)
            result_focused = tool._run(
                image_path=str(image_path), focus="color and style"
            )
            print(f"Focused Analysis:\n{result_focused}")
            print("\n✅ Test passed")

        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            logger.error(f"Error in VLM test case {idx}", exc_info=True)


def test_tool_with_langchain():
    """Test tool integration with LangChain"""
    print("\n" + "=" * 80)
    print("TEST 5: LangChain Tool Integration".center(80))
    print("=" * 80 + "\n")

    try:
        from langchain.agents import AgentType, initialize_agent
        from langchain_openai import ChatOpenAI

        from app.config import settings

        # Initialize LLM
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            api_key=settings.openai_api_key,
        )

        # Initialize tools
        tools = [
            ProductSearchTool(),
            FilterProductsTool(),
            ImageSearchTool(),
            VLMReasoningTool(),
        ]

        # Create agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=3,
        )

        # Test queries
        test_queries = [
            "Find me red dresses for women",
            "Show me blue shoes",
        ]

        for query in test_queries:
            print(f"\n{'=' * 80}")
            print(f"Query: {query}")
            print(f"{'=' * 80}\n")

            try:
                response = agent.run(query)
                print(f"\nAgent Response:\n{response}\n")
                print("✅ Agent test passed")
            except Exception as e:
                print(f"\n❌ Agent test failed: {e}")
                logger.error(f"Error in agent test: {query}", exc_info=True)

    except ImportError as e:
        print(f"⚠️  Skipping LangChain integration test: {e}")
        print("   Install langchain-openai: pip install langchain-openai")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("LANGCHAIN TOOLS TEST SUITE".center(80))
    print("=" * 80)

    try:
        # Test 1: Product Search
        test_product_search()

        # Test 2: Filter Products
        test_filter_products()

        # Test 3: Image Search
        test_image_search()

        # Test 4: VLM Reasoning
        test_vlm_reasoning()

        # Test 5: LangChain Integration (optional)
        test_tool_with_langchain()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED".center(80))
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
    except Exception as e:
        logger.error("Unexpected error in test suite", exc_info=True)
        print(f"\n❌ Test suite failed: {e}")


if __name__ == "__main__":
    main()
