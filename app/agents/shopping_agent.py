"""
Conversational Shopping Agent
Implements Flows 1-4 with tool orchestration
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.tools import BaseTool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

from app.agents.flow_router import FlowRouter, FlowType
from app.config import settings
from app.tools import (
    FilterProductsTool,
    ImageSearchTool,
    ProductSearchTool,
    VLMReasoningTool,
)

logger = logging.getLogger(__name__)


class ShoppingAgent:
    """Conversational shopping agent with tool orchestration"""

    def __init__(
        self,
        session_id: Optional[str] = None,
        chat_history: Optional[ChatMessageHistory] = None,
        llm: Optional[ChatOpenAI] = None,
        tools: Optional[Dict[str, BaseTool]] = None,
    ):
        """Initialize shopping agent

        Args:
            session_id: Unique session identifier for conversation tracking
            chat_history: Optional pre-configured chat history (for persistent conversations)
        """
        self.session_id = session_id or "default"
        self.llm = llm or ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            api_key=settings.openai_api_key,
        )

        # Initialize tools
        self.tools = tools or {
            "search_products": ProductSearchTool(),
            "filter_products": FilterProductsTool(),
            "search_by_image": ImageSearchTool(),
            "analyze_image_style": VLMReasoningTool(),
        }

        # Initialize flow router
        self.flow_router = FlowRouter(llm=self.llm)

        # Initialize chat history
        self.chat_history = chat_history or ChatMessageHistory()
        self.current_step = ""  # Track current processing step
        self.tools_used = []  # Track tools used in current query

        logger.info(f"Shopping agent initialized for session: {self.session_id}")

    def chat(
        self,
        query: str,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process user query and return response

        Args:
            query: User's text query
            image_path: Optional path to uploaded image

        Returns:
            Dict with 'response' and optional 'error'
        """
        try:
            logger.info(
                f"[{self.session_id}] Processing query: '{query}' (image={'Yes' if image_path else 'No'})"
            )

            # Validate image if provided
            if image_path and not Path(image_path).exists():
                return {
                    "response": f"Error: Image file not found at '{image_path}'",
                    "error": True,
                }

            # Build context from chat history
            context = self._build_context(self.chat_history.messages)

            # Route query to appropriate flow
            response_text = self._route_and_execute(query, image_path, context)

            # Save to chat history
            self.chat_history.add_user_message(query)
            self.chat_history.add_ai_message(response_text)

            logger.info(f"[{self.session_id}] Response generated successfully")

            return {
                "response": response_text,
                "error": False,
            }

        except Exception as e:
            logger.error(f"Error in agent chat: {e}", exc_info=True)
            return {
                "response": f"I apologize, I encountered an error: {str(e)}",
                "error": True,
            }

    def _build_context(self, chat_history: List) -> str:
        """Build conversation context"""
        if not chat_history:
            return "No previous conversation."

        context_lines = []
        for msg in chat_history[-6:]:  # Last 3 exchanges
            if hasattr(msg, "type"):
                role = "User" if msg.type == "human" else "Assistant"
                content = msg.content[:150]
                context_lines.append(f"{role}: {content}")

        return "\n".join(context_lines)

    def _route_and_execute(
        self,
        query: str,
        image_path: Optional[str],
        context: str,
    ) -> str:
        """Route query to appropriate tools and generate response using LLM-based routing"""

        # Use FlowRouter to determine the best flow
        flow_type = self.flow_router.route(
            query=query, has_image=bool(image_path), context=context
        )

        logger.info(f"Executing flow: {flow_type.value}")

        # Execute the appropriate flow
        if flow_type == FlowType.TEXT_RAG:
            return self._execute_text_search(query, context)
        elif flow_type == FlowType.VISUAL_SEARCH:
            return self._execute_visual_search(query, image_path, context)
        elif flow_type == FlowType.VISUAL_FILTER:
            return self._execute_visual_with_filter(query, image_path, context)
        elif flow_type == FlowType.VLM_ANALYSIS:
            return self._execute_vlm_only(query, image_path, context)
        elif flow_type == FlowType.VLM_SEARCH:
            return self._execute_vlm_then_search(query, image_path, context)
        else:
            # Fallback
            logger.warning(f"Unknown flow type: {flow_type}, using text search")
            return self._execute_text_search(query, context)

    def _execute_text_search(self, query: str, context: str) -> str:
        """Flow 1: Text-based RAG search"""
        logger.info("Executing Flow 1: Text RAG Search")
        self.tools_used = []  # Reset tools tracking

        # If there's context, use LLM to create a better search query
        search_query = query
        if context and context != "No previous conversation.":
            self.current_step = "🤔 Understanding context..."
            self.tools_used.append("Context Understanding")
            context_prompt = f"""Based on the conversation history and user's request, create a concise search query.

Previous conversation:
{context}

User's current request: {query}

Create a search query (2-8 words) that captures what the user wants, considering the context.
For example:
- If user says "find pants for this shirt" and previous context mentions "black graphic t-shirt", return "white pants casual"
- If user asks "what about in red?", return the item type from context + "red"

ONLY return the search query, nothing else."""

            query_response = self.llm.invoke(context_prompt)
            search_query = query_response.content.strip()
            logger.info(f"Enhanced search query: {search_query}")

        self.current_step = "🔍 Searching for products..."
        self.tools_used.append("Text Search (RAG)")
        tool = self.tools["search_products"]
        result = tool._run(query=search_query, limit=5)

        self.current_step = "✨ Preparing results..."

        # Generate natural language response
        prompt = f"""You are a helpful fashion shopping assistant.

Previous conversation:
{context}

User query: {query}

Search results (searched for: "{search_query}"):
{result}

Based on the search results, provide a friendly response to the user.

IMPORTANT: You MUST include ALL product details in your response using this EXACT format for each product:

1. [Product Name]
   ID: [Product ID Number]
   Category: [Category details]
   Color: [Color]
   Gender: [Gender]
   Season: [Season if available]
   Usage: [Usage if available]
   Relevance: [Relevance percentage if available]

Keep the friendly tone but preserve ALL the product information exactly as provided, especially the ID field."""

        response = self.llm.invoke(prompt)
        return response.content

    def _execute_visual_search(self, query: str, image_path: str, context: str) -> str:
        """Flow 2: Pure visual search"""
        logger.info("Executing Flow 2: Visual Search")
        self.tools_used = ["Image Search (CLIP)"]  # Reset and track

        self.current_step = "🖼️ Analyzing image..."
        tool = self.tools["search_by_image"]
        result = tool._run(image_path=image_path, limit=5)

        self.current_step = "✨ Preparing results..."

        prompt = f"""You are a helpful fashion shopping assistant.

Previous conversation:
{context}

User uploaded an image and asked: {query}

Visually similar products found:
{result}

Present these similar products to the user in a friendly way.

IMPORTANT: You MUST include ALL product details in your response using this EXACT format for each product:

1. [Product Name]
   ID: [Product ID Number]
   Category: [Category details]
   Color: [Color]
   Gender: [Gender]
   Season: [Season if available]
   Usage: [Usage if available]
   Similarity: [Similarity percentage if available]

Keep the friendly tone but preserve ALL the product information exactly as provided, especially the ID field."""

        response = self.llm.invoke(prompt)
        return response.content

    def _execute_visual_with_filter(
        self, query: str, image_path: str, context: str
    ) -> str:
        """Flow 3: Visual search + attribute filtering"""
        logger.info("Executing Flow 3: Visual Search + Filter")
        self.tools_used = ["Image Search (CLIP)", "Metadata Filter"]

        self.current_step = "🖼️ Analyzing image..."
        visual_tool: ImageSearchTool = self.tools["search_by_image"]  # type: ignore

        try:
            raw_results = visual_tool.search(image_path=image_path, limit=50)
        except FileNotFoundError as exc:
            return f"I could not find the uploaded image: {exc}"
        except Exception as exc:  # Catch embedding or search issues
            logger.error(f"Visual search failed: {exc}", exc_info=True)
            return "I ran into an issue while analyzing the image. Could you try again later?"

        if not raw_results:
            return "I could not find similar items for that image."

        self.current_step = "🎯 Applying filters..."

        filters = self._extract_filters(query)
        logger.info(f"Extracted filters from query: {filters}")

        filtered_results = self._apply_metadata_filters(raw_results, filters)

        filter_note = ""
        if not filtered_results:
            filter_note = "I could not find results that match those specific filters. Here are the closest visual matches instead."
            filtered_results = raw_results[:5]

        formatted_results = visual_tool._format_results(filtered_results, image_path)  # type: ignore

        response_lines = [
            "Here are the closest matches based on your image.",
        ]

        if filters:
            response_lines.append(
                "Filters applied: "
                + ", ".join(f"{key}={value}" for key, value in filters.items())
            )

        if filter_note:
            response_lines.append(filter_note)

        response_lines.append("")
        response_lines.append(formatted_results)

        return "\n".join(line for line in response_lines if line).strip()

    def _execute_vlm_only(self, query: str, image_path: str, context: str) -> str:
        """VLM analysis only - no product search"""
        logger.info("Executing VLM Analysis Only")
        self.tools_used = ["VLM Analysis (GPT-4o-mini)"]  # Only VLM

        self.current_step = "👁️ Analyzing image with AI..."
        vlm_tool = self.tools["analyze_image_style"]
        style_analysis = vlm_tool._run(image_path=image_path)

        self.current_step = "✨ Preparing response..."

        # Generate response with only style analysis
        prompt = f"""You are a helpful fashion shopping assistant.

Previous conversation:
{context}

User uploaded an image and asked: {query}

Image Analysis:
{style_analysis}

Based on the analysis, provide a friendly and informative response about the style.
DO NOT recommend products unless the user specifically asked for recommendations.
Just describe what you see in the image and answer the user's question."""

        response = self.llm.invoke(prompt)
        return response.content

    def _execute_vlm_then_search(
        self, query: str, image_path: str, context: str
    ) -> str:
        """Flow 4: ReAct Loop - VLM analysis then text search"""
        logger.info("Executing Flow 4: ReAct Loop (VLM -> Text Search)")
        self.tools_used = [
            "VLM Analysis (GPT-4o-mini)",
            "Text Search (RAG)",
        ]  # Reset and track

        # Step 1: Analyze image with VLM
        self.current_step = "👁️ Analyzing image style with AI..."
        vlm_tool = self.tools["analyze_image_style"]
        style_analysis = vlm_tool._run(image_path=image_path)

        logger.info(f"VLM Analysis: {style_analysis}")

        # Step 2: Extract search query from analysis and user intent
        self.current_step = "🔍 Searching based on style..."
        search_prompt = f"""Based on the image analysis and user's request, create a search query.

User request: {query}
Image analysis: {style_analysis}

Create a concise search query (2-5 words) that captures what the user wants.
ONLY return the search query, nothing else."""

        search_query_response = self.llm.invoke(search_prompt)
        search_query = search_query_response.content.strip()

        logger.info(f"Generated search query: {search_query}")

        # Step 3: Search with the generated query
        search_tool = self.tools["search_products"]
        search_results = search_tool._run(query=search_query, limit=5)

        # Step 4: Generate final response
        self.current_step = "✨ Preparing results..."
        final_prompt = f"""You are a helpful fashion shopping assistant.

Previous conversation:
{context}

User uploaded an image and asked: {query}

Step 1 - Image Analysis:
{style_analysis}

Step 2 - Search Results for "{search_query}":
{search_results}

Provide a comprehensive response that:
1. Acknowledges the image style
2. Presents the search results
3. Explains how they match the user's request

IMPORTANT: You MUST include ALL product details in your response using this EXACT format for each product:

1. [Product Name]
   ID: [Product ID Number]
   Category: [Category details]
   Color: [Color]
   Gender: [Gender]
   Season: [Season if available]
   Usage: [Usage if available]
   Relevance: [Relevance percentage if available]

Keep the friendly tone but preserve ALL the product information exactly as provided, especially the ID field."""

        response = self.llm.invoke(final_prompt)
        return response.content

    def _extract_filters(self, query: str) -> Dict[str, str]:
        """Extract structured filters from user query"""

        filter_prompt = f"""Extract filtering criteria from this query: "{query}"

Return ONLY a JSON object with these optional fields (omit fields that aren't mentioned):
- gender (Men/Women/Boys/Girls/Unisex)
- baseColour (color name)
- masterCategory (Apparel/Accessories/Footwear)
- season (Summer/Winter/Fall/Spring)

Query: {query}
JSON:"""

        try:
            response = self.llm.invoke(filter_prompt)
            raw_response = (
                response.content.strip() if response and response.content else "{}"
            )

            start = raw_response.find("{")
            end = raw_response.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_text = raw_response[start : end + 1]
            else:
                json_text = "{}"

            parsed = json.loads(json_text)
            if not isinstance(parsed, dict):
                return {}

            # Normalize values (title case for readability)
            normalized = {
                key: str(value).strip() for key, value in parsed.items() if value
            }

            return normalized

        except Exception as exc:
            logger.warning(f"Failed to extract filters: {exc}")
            return {}

    def _apply_metadata_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, str],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Apply metadata filters (case-insensitive) to search results"""

        if not filters:
            return results[:limit]

        def matches(item: Dict[str, Any]) -> bool:
            for key, expected in filters.items():
                mapped_key = self._map_filter_key(key)
                if not mapped_key:
                    continue
                value = item.get(mapped_key)
                if value is None:
                    return False
                if str(value).strip().lower() != expected.strip().lower():
                    return False
            return True

        filtered: List[Dict[str, Any]] = []
        for item in results:
            if matches(item):
                filtered.append(item)
            if len(filtered) >= limit:
                break

        return filtered

    def _map_filter_key(self, key: str) -> Optional[str]:
        """Map filter keys from LLM to result metadata keys"""

        mapping = {
            "gender": "gender",
            "basecolour": "baseColour",
            "basecolor": "baseColour",
            "color": "baseColour",
            "colour": "baseColour",
            "mastercategory": "masterCategory",
            "category": "masterCategory",
            "season": "season",
        }

        return mapping.get(key.lower())

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history for this session"""
        try:
            messages = self.chat_history.messages
            result = []
            for msg in messages:
                if hasattr(msg, "type"):
                    role = "user" if msg.type == "human" else "assistant"
                    result.append({"role": role, "content": msg.content})
            return result
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    def clear_history(self):
        """Clear conversation history for this session"""
        self.chat_history.clear()
        logger.info(f"[{self.session_id}] Conversation history cleared")


def create_shopping_agent(session_id: Optional[str] = None) -> ShoppingAgent:
    """Factory function to create a shopping agent

    Args:
        session_id: Unique session identifier

    Returns:
        Configured ShoppingAgent instance
    """
    return ShoppingAgent(session_id=session_id)
