"""
Conversational Shopping Agent
Implements Flows 1-4 with tool orchestration
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

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
    ):
        """Initialize shopping agent

        Args:
            session_id: Unique session identifier for conversation tracking
            chat_history: Optional pre-configured chat history (for persistent conversations)
        """
        self.session_id = session_id or "default"
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            api_key=settings.openai_api_key,
        )

        # Initialize tools
        self.tools = {
            "search_products": ProductSearchTool(),
            "filter_products": FilterProductsTool(),
            "search_by_image": ImageSearchTool(),
            "analyze_image_style": VLMReasoningTool(),
        }

        # Initialize chat history
        self.chat_history = chat_history or ChatMessageHistory()

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
        """Route query to appropriate tools and generate response"""

        query_lower = query.lower()

        # Determine which tool(s) to use
        if image_path:
            # Image-based queries
            if any(word in query_lower for word in ["analyze", "describe", "what is", "tell me about", "style"]):
                # Flow 4: VLM analysis first, then search
                return self._execute_vlm_then_search(query, image_path, context)
            elif any(word in query_lower for word in ["but", "in", "with", "filter"]):
                # Flow 3: Visual search + filter
                return self._execute_visual_with_filter(query, image_path, context)
            else:
                # Flow 2: Pure visual search
                return self._execute_visual_search(query, image_path, context)
        else:
            # Text-only queries
            if any(word in query_lower for word in ["filter", "gender", "color", "season", "category"]):
                # Use filter tool
                return self._execute_filter(query, context)
            else:
                # Flow 1: Text RAG search
                return self._execute_text_search(query, context)

    def _execute_text_search(self, query: str, context: str) -> str:
        """Flow 1: Text-based RAG search"""
        logger.info("Executing Flow 1: Text RAG Search")

        tool = self.tools["search_products"]
        result = tool._run(query=query, limit=5)

        # Generate natural language response
        prompt = f"""You are a helpful fashion shopping assistant.

Previous conversation:
{context}

User query: {query}

Search results:
{result}

Based on the search results, provide a friendly response to the user. Present the products clearly and mention key details like color, category, and style."""

        response = self.llm.invoke(prompt)
        return response.content

    def _execute_visual_search(self, query: str, image_path: str, context: str) -> str:
        """Flow 2: Pure visual search"""
        logger.info("Executing Flow 2: Visual Search")

        tool = self.tools["search_by_image"]
        result = tool._run(image_path=image_path, limit=5)

        prompt = f"""You are a helpful fashion shopping assistant.

Previous conversation:
{context}

User uploaded an image and asked: {query}

Visually similar products found:
{result}

Present these similar products to the user in a friendly way. Mention the visual similarity scores and key product details."""

        response = self.llm.invoke(prompt)
        return response.content

    def _execute_visual_with_filter(self, query: str, image_path: str, context: str) -> str:
        """Flow 3: Visual search + attribute filtering"""
        logger.info("Executing Flow 3: Visual Search + Filter")

        # First get visually similar items
        visual_tool = self.tools["search_by_image"]
        visual_results = visual_tool._run(image_path=image_path, limit=10)

        # Extract filter criteria from query
        filter_prompt = f"""Extract filtering criteria from this query: "{query}"

Return ONLY a JSON object with these optional fields (omit fields that aren't mentioned):
- gender (Men/Women/Boys/Girls/Unisex)
- baseColour (color name)
- masterCategory (Apparel/Accessories/Footwear)
- season (Summer/Winter/Fall/Spring)

Query: {query}
JSON:"""

        filter_response = self.llm.invoke(filter_prompt)
        
        # For now, just use visual results with LLM filtering
        prompt = f"""You are a helpful fashion shopping assistant.

Previous conversation:
{context}

User uploaded an image and asked: {query}

Visually similar products found:
{visual_results}

Filter the results based on the user's additional requirements (e.g., color, gender) and present the best matches."""

        response = self.llm.invoke(prompt)
        return response.content

    def _execute_vlm_then_search(self, query: str, image_path: str, context: str) -> str:
        """Flow 4: ReAct Loop - VLM analysis then text search"""
        logger.info("Executing Flow 4: ReAct Loop (VLM -> Text Search)")

        # Step 1: Analyze image with VLM
        vlm_tool = self.tools["analyze_image_style"]
        style_analysis = vlm_tool._run(image_path=image_path)

        logger.info(f"VLM Analysis: {style_analysis}")

        # Step 2: Extract search query from analysis and user intent
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
3. Explains how they match the user's request"""

        response = self.llm.invoke(final_prompt)
        return response.content

    def _execute_filter(self, query: str, context: str) -> str:
        """Execute attribute-based filtering"""
        logger.info("Executing attribute filter")

        # Use LLM to extract filter parameters
        filter_prompt = f"""Extract filter parameters from this query: "{query}"

Return ONLY a JSON object with these optional fields (omit if not mentioned):
- gender (Men/Women/Boys/Girls/Unisex)
- baseColour (color name)
- masterCategory (Apparel/Accessories/Footwear/Personal Care)
- subCategory (Topwear/Bottomwear/Shoes/Bags/etc)
- season (Summer/Winter/Fall/Spring)

Query: {query}
JSON:"""

        # For simplicity, fallback to text search for now
        return self._execute_text_search(query, context)

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
