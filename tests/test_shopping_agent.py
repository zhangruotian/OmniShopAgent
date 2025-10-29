import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("OPENAI_API_KEY", "test-key")


class StubLLM:
    def __init__(self, responses=None):
        self.responses = responses or []
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        content = self.responses.pop(0) if self.responses else "{}"
        return SimpleNamespace(content=content)


class StubImageSearchTool:
    def __init__(self, results):
        self.results = results
        self.search_calls = []
        self.run_calls = []

    def search(self, image_path, limit=5, filters=None):
        self.search_calls.append((image_path, limit, filters))
        return self.results

    def _run(self, image_path, limit=5, filters=None):
        self.run_calls.append((image_path, limit, filters))
        formatted = self._format_results(self.results[:limit], image_path)
        return formatted or "No similar products found."

    def _format_results(self, results, image_path):
        blocks = []
        for idx, item in enumerate(results, 1):
            blocks.append(f"{idx}. {item.get('productDisplayName')}")
            blocks.append(f"   ID: {item.get('id')}")
            blocks.append(
                "   Category: "
                f"{item.get('masterCategory')} > {item.get('subCategory')} > {item.get('articleType')}"
            )
            blocks.append(f"   Color: {item.get('baseColour')}")
            blocks.append(f"   Gender: {item.get('gender')}")
            if item.get("season"):
                blocks.append(f"   Season: {item.get('season')}")
            if item.get("usage"):
                blocks.append(f"   Usage: {item.get('usage')}")
            if item.get("distance") is not None:
                similarity = 1 - item.get("distance")
                blocks.append(f"   Visual Similarity: {similarity:.2%}")
            blocks.append("")
        return "\n".join(line for line in blocks if line)


class StubProductSearchTool:
    def __init__(self, result=""):
        self.result = result
        self.calls = []

    def _run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class StubFilterTool:
    def _run(self, *args, **kwargs):
        return ""


class StubVLMTool:
    def __init__(self, result=""):
        self.result = result
        self.calls = []

    def _run(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


def build_agent(stub_image_tool, responses, product_tool=None, vlm_tool=None):
    from langchain_community.chat_message_histories import ChatMessageHistory

    from app.agents.shopping_agent import ShoppingAgent

    product_tool = product_tool or StubProductSearchTool()
    vlm_tool = vlm_tool or StubVLMTool()

    tools = {
        "search_products": product_tool,
        "filter_products": StubFilterTool(),
        "search_by_image": stub_image_tool,
        "analyze_image_style": vlm_tool,
    }

    llm = StubLLM(responses=responses)
    history = ChatMessageHistory()

    return ShoppingAgent(
        session_id="test-session",
        chat_history=history,
        llm=llm,
        tools=tools,
    )


@pytest.fixture
def sample_results():
    return [
        {
            "id": 1,
            "productDisplayName": "Azure Summer Dress",
            "masterCategory": "Apparel",
            "subCategory": "Dress",
            "articleType": "Casual Dress",
            "baseColour": "Blue",
            "gender": "Women",
            "season": "Summer",
            "usage": "Casual",
            "distance": 0.1,
        },
        {
            "id": 2,
            "productDisplayName": "Crimson Evening Gown",
            "masterCategory": "Apparel",
            "subCategory": "Dress",
            "articleType": "Formal Dress",
            "baseColour": "Red",
            "gender": "Women",
            "season": "Winter",
            "usage": "Party",
            "distance": 0.2,
        },
    ]


def create_temp_image(tmp_path):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"test")
    return image_path


def test_visual_filter_applies_metadata(sample_results, tmp_path):
    stub_tool = StubImageSearchTool(sample_results)
    agent = build_agent(
        stub_tool,
        responses=[
            '{"flow": "visual_filter", "reasoning": "User wants filters"}',  # FlowRouter
            '{"gender": "Women", "baseColour": "Blue"}',  # Extract filters
        ],
    )

    image_path = create_temp_image(tmp_path)

    result = agent.chat(
        query="Find similar items but only the blue option for women",
        image_path=str(image_path),
    )

    assert "Azure Summer Dress" in result["response"]
    assert "Crimson Evening Gown" not in result["response"]
    assert "Filters applied" in result["response"]
    assert agent.tools_used == ["Image Search (CLIP)", "Metadata Filter"]


def test_visual_filter_fallback_to_visual(sample_results, tmp_path):
    stub_tool = StubImageSearchTool(sample_results)
    agent = build_agent(
        stub_tool,
        responses=[
            '{"flow": "visual_filter", "reasoning": "User wants filters"}',  # FlowRouter
            '{"gender": "Men"}',  # Extract filters
        ],
    )

    image_path = create_temp_image(tmp_path)

    result = agent.chat(
        query="Show me similar items but for men",
        image_path=str(image_path),
    )

    assert "Azure Summer Dress" in result["response"]
    assert "I could not find results that match" in result["response"]


def test_visual_filter_handles_invalid_json(sample_results, tmp_path):
    stub_tool = StubImageSearchTool(sample_results)
    agent = build_agent(
        stub_tool,
        responses=[
            '{"flow": "visual_search", "reasoning": "Fallback"}',  # FlowRouter
            "Here are similar items",  # Final response
        ],
    )

    image_path = create_temp_image(tmp_path)

    result = agent.chat(
        query="Find similar items with some filters",
        image_path=str(image_path),
    )

    assert "Here are similar items" in result["response"]


def test_chat_returns_error_for_missing_image():
    stub_tool = StubImageSearchTool([])
    agent = build_agent(stub_tool, responses=["{}"])

    result = agent.chat(query="Find similar items", image_path="/nonexistent.jpg")

    assert result["error"] is True
    assert "Image file not found" in result["response"]


def test_visual_similarity_prefers_image_search(sample_results, tmp_path):
    stub_tool = StubImageSearchTool(sample_results)
    agent = build_agent(
        stub_tool,
        responses=[
            '{"flow": "visual_search", "reasoning": "Direct visual similarity"}',  # FlowRouter
            "Here are some similar items",  # Final response
        ],
    )

    image_path = create_temp_image(tmp_path)

    result = agent.chat(
        query="Find me ones visually similar",
        image_path=str(image_path),
    )

    assert agent.tools_used == ["Image Search (CLIP)"]
    assert stub_tool.run_calls, "Image search tool should be invoked"
    assert "Here are some similar items" in result["response"]


def test_product_search_tool_initialization(monkeypatch):
    from app.tools import search_tools

    class DummyEmbeddingService:
        def __init__(self, *args, **kwargs):
            pass

        def connect_clip(self):
            raise AssertionError("connect_clip should not be called")

    class DummyMilvusService:
        def __init__(self, *args, **kwargs):
            self.connected = False

        def connect(self):
            self.connected = True

        def is_connected(self):
            return self.connected

        def search_similar_text(self, *args, **kwargs):
            return []

    monkeypatch.setattr(search_tools, "EmbeddingService", DummyEmbeddingService)
    monkeypatch.setattr(search_tools, "MilvusService", DummyMilvusService)

    tool = search_tools.ProductSearchTool()

    assert isinstance(tool.embedding_service, DummyEmbeddingService)
    assert tool.milvus_service.is_connected()


def test_new_request_with_image_uses_text_search(sample_results, tmp_path):
    stub_image_tool = StubImageSearchTool(sample_results)
    product_tool = StubProductSearchTool(result="Search results")
    vlm_tool = StubVLMTool(result="Style analysis")

    responses = [
        '{"flow": "vlm_analysis", "reasoning": "Description only"}',  # FlowRouter for first query
        "Style description",  # First turn (VLM-only response)
        '{"flow": "vlm_search", "reasoning": "Style-based search"}',  # FlowRouter for second query
        "white pants",  # Generated search query for text search
        "Here are some pants",  # Final assistant response
    ]

    agent = build_agent(
        stub_image_tool,
        responses=responses,
        product_tool=product_tool,
        vlm_tool=vlm_tool,
    )

    image_path = create_temp_image(tmp_path)

    first_result = agent.chat(
        query="What is the style of this shirt?",
        image_path=str(image_path),
    )
    assert "Style description" in first_result["response"]

    second_result = agent.chat(
        query="Find me a white pants",
        image_path=str(image_path),
    )

    assert (
        "VLM Analysis (GPT-4o-mini)" in agent.tools_used
        or "Text Search (RAG)" in agent.tools_used
    )
    assert product_tool.calls, "Search tool should be used"
    assert "Here are some pants" in second_result["response"]
