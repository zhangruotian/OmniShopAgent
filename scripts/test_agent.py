"""
Test script for Agent Layer
Tests all flows: Intent Classification, Shopping Agent, and Orchestrator
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents import (
    IntentType,
    create_shopping_agent,
    get_boundary_handler,
    get_intent_classifier,
)
from app.agents.orchestrator import AgentOrchestrator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")


def test_intent_classifier():
    """Test Flow 0: Intent Classification"""
    print_section("TEST 1: Intent Classification (Flow 0)")

    classifier = get_intent_classifier()
    handler = get_boundary_handler()

    test_cases = [
        # Specific search queries
        ("Find me blue summer dresses", False, IntentType.SPECIFIC_SEARCH),
        ("Show me red running shoes for men", False, IntentType.SPECIFIC_SEARCH),
        ("I need formal wear", True, IntentType.SPECIFIC_SEARCH),
        # Too vague
        ("Recommend something", False, IntentType.TOO_VAGUE),
        ("I want to buy something", False, IntentType.TOO_VAGUE),
        # Out of scope
        ("I want to buy a phone", False, IntentType.OUT_OF_SCOPE),
        ("Recommend some furniture", False, IntentType.OUT_OF_SCOPE),
        # Chitchat
        ("Hello", False, IntentType.CHITCHAT),
        ("Thank you", False, IntentType.CHITCHAT),
    ]

    passed = 0
    for query, has_image, expected_intent in test_cases:
        print(f"Query: '{query}' (image={has_image})")
        print("-" * 80)

        classification = classifier.classify(query, has_image)
        print(f"Intent: {classification.intent.value}")
        print(f"Reasoning: {classification.reasoning}")

        if classification.intent != IntentType.SPECIFIC_SEARCH:
            response = handler.handle(classification, query, has_image)
            print(f"\nBoundary Response:\n{response}")

        # Check if classification matches expected
        if classification.intent == expected_intent:
            print("\n✅ Test passed\n")
            passed += 1
        else:
            print(f"\n❌ Test failed (expected {expected_intent.value})\n")

    print(f"\n📊 Results: {passed}/{len(test_cases)} tests passed\n")


def test_shopping_agent():
    """Test Flows 1-4: Shopping Agent"""
    print_section("TEST 2: Shopping Agent (Flows 1-4)")

    agent = create_shopping_agent(session_id="test_session")

    # Get sample image
    image_dir = Path("data/images")
    sample_images = list(image_dir.glob("*.jpg"))[:2] if image_dir.exists() else []

    test_cases = [
        # Flow 1: Text RAG
        {
            "name": "Flow 1: Text RAG Search",
            "query": "Show me casual blue shirts for men",
            "image": None,
        },
        # Flow 2: Visual Search (if image available)
        {
            "name": "Flow 2: Visual Search",
            "query": "Find similar products",
            "image": str(sample_images[0]) if sample_images else None,
        },
        # Flow 4: ReAct Loop (VLM + Text Search)
        {
            "name": "Flow 4: ReAct Loop (VLM analysis)",
            "query": "Analyze the style of this product and find similar items",
            "image": str(sample_images[1]) if len(sample_images) > 1 else None,
        },
    ]

    for idx, test_case in enumerate(test_cases, 1):
        if test_case["image"] and not Path(test_case["image"]).exists():
            print(f"⚠️  Skipping {test_case['name']}: No image available\n")
            continue

        print(f"\nTest Case {idx}: {test_case['name']}")
        print("-" * 80)
        print(f"Query: {test_case['query']}")
        if test_case["image"]:
            print(f"Image: {Path(test_case['image']).name}")
        print()

        try:
            result = agent.chat(
                query=test_case["query"],
                image_path=test_case["image"],
            )

            print(f"Response:\n{result['response']}\n")
            print("✅ Test completed\n")

        except Exception as e:
            print(f"❌ Test failed: {e}\n")
            logger.error(f"Error in test case {idx}", exc_info=True)


def test_orchestrator():
    """Test Full Orchestrator (All Flows)"""
    print_section("TEST 3: Agent Orchestrator (All Flows)")

    orchestrator = AgentOrchestrator(session_id="test_orchestrator")

    # Get sample image
    image_dir = Path("data/images")
    sample_image = None
    if image_dir.exists():
        images = list(image_dir.glob("*.jpg"))
        if images:
            sample_image = str(images[0])

    test_cases = [
        # Chitchat
        {"query": "Hello!", "image": None, "expected_flow": "flow_0_boundary"},
        # Out of scope
        {
            "query": "I want to buy a laptop",
            "image": None,
            "expected_flow": "flow_0_boundary",
        },
        # Specific search
        {
            "query": "Find me red casual dresses",
            "image": None,
            "expected_flow": "flow_1_text_rag",
        },
        # Visual search
        {
            "query": "Find similar items",
            "image": sample_image,
            "expected_flow": "flow_2_visual",
        },
    ]

    passed = 0
    for idx, test_case in enumerate(test_cases, 1):
        if test_case["image"] and not Path(test_case["image"]).exists():
            print(f"⚠️  Skipping test {idx}: No image available\n")
            continue

        print(f"\nTest Case {idx}: {test_case['query']}")
        print("-" * 80)

        try:
            result = orchestrator.process_query(
                query=test_case["query"],
                image_path=test_case["image"],
            )

            print(f"Intent: {result['intent']}")
            print(f"Flow: {result['flow']}")
            print(f"Response:\n{result['response']}\n")

            if result["flow"] == test_case["expected_flow"]:
                print("✅ Flow matched\n")
                passed += 1
            else:
                print(
                    f"⚠️  Flow mismatch (expected {test_case['expected_flow']})\n"
                )

        except Exception as e:
            print(f"❌ Test failed: {e}\n")
            logger.error(f"Error in test case {idx}", exc_info=True)

    print(f"\n📊 Results: {passed}/{len(test_cases)} tests passed flow expectation\n")


def test_conversation_flow():
    """Test Flow 5: Conversational Memory"""
    print_section("TEST 4: Conversational Memory (Flow 5)")

    orchestrator = AgentOrchestrator(session_id="memory_test")

    conversation = [
        "Show me blue casual shirts for men",
        "What about in red color?",  # Should remember previous context
        "Find matching shoes",  # Should remember the style context
    ]

    for idx, query in enumerate(conversation, 1):
        print(f"\nTurn {idx}: {query}")
        print("-" * 80)

        result = orchestrator.process_query(query)
        print(f"Response:\n{result['response']}\n")

    print("\nConversation History:")
    print("-" * 80)
    history = orchestrator.get_conversation_history()
    for msg in history[-6:]:  # Show last 3 exchanges
        print(f"{msg['role'].upper()}: {msg['content'][:100]}...")

    print("\n✅ Conversation test completed\n")


if __name__ == "__main__":
    print("=" * 80)
    print("AGENT LAYER TEST SUITE".center(80))
    print("=" * 80)

    try:
        # Test 1: Intent Classification
        test_intent_classifier()

        # Test 2: Shopping Agent
        test_shopping_agent()

        # Test 3: Orchestrator
        test_orchestrator()

        # Test 4: Conversational Memory
        test_conversation_flow()

        print("=" * 80)
        print("ALL AGENT TESTS COMPLETED".center(80))
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        logger.error("Fatal error in test suite", exc_info=True)
        print(f"\n❌ Test suite failed: {e}")

