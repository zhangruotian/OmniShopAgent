"""
Test script for LangChain tools
Tests product search and filter functionality
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tools import FilterProductsTool, ProductSearchTool

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


def test_tool_with_langchain():
    """Test tool integration with LangChain"""
    print("\n" + "=" * 80)
    print("TEST 3: LangChain Tool Integration".center(80))
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
        tools = [ProductSearchTool(), FilterProductsTool()]

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

        # Test 3: LangChain Integration (optional)
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

