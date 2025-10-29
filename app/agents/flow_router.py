"""
LLM-based Flow Router
Intelligently routes queries to appropriate execution flows
"""

import json
import logging
from enum import Enum
from typing import Optional

from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class FlowType(str, Enum):
    """Available execution flows"""

    TEXT_RAG = "text_rag"  # Flow 1: Text search
    VISUAL_SEARCH = "visual_search"  # Flow 2: Pure image similarity
    VISUAL_FILTER = "visual_filter"  # Flow 3: Image + filters
    VLM_ANALYSIS = "vlm_analysis"  # Flow 4a: Only describe image
    VLM_SEARCH = "vlm_search"  # Flow 4b: Analyze + search


class FlowRouter:
    """LLM-based intelligent flow router"""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """Initialize flow router

        Args:
            llm: Optional language model (uses default if not provided)
        """
        self.llm = llm or ChatOpenAI(
            model=settings.openai_model,
            temperature=0.1,  # Low temperature for consistent routing
            api_key=settings.openai_api_key,
        )

    def route(
        self, query: str, has_image: bool, context: str = "No previous conversation."
    ) -> FlowType:
        """Determine the best flow using LLM

        Args:
            query: User's query text
            has_image: Whether user uploaded an image
            context: Conversation context

        Returns:
            FlowType enum indicating which flow to use
        """
        try:
            # If no image, always use text search
            if not has_image:
                logger.info("No image provided, routing to TEXT_RAG")
                return FlowType.TEXT_RAG

            # For image queries, use LLM to decide
            prompt = self._build_routing_prompt(query, context)
            response = self.llm.invoke(prompt)
            flow_type = self._parse_response(response.content)

            logger.info(f"LLM routed to: {flow_type.value}")
            return flow_type

        except Exception as e:
            logger.error(f"Error in flow routing: {e}", exc_info=True)
            # Safe fallback
            if has_image:
                return FlowType.VISUAL_SEARCH
            return FlowType.TEXT_RAG

    def _build_routing_prompt(self, query: str, context: str) -> str:
        """Build prompt for flow routing"""
        return f"""You are a routing system for a fashion shopping agent. Choose the BEST flow for this query.

AVAILABLE FLOWS:
1. TEXT_RAG - Text-based product search (IGNORE the uploaded image)
   Use when: User requests a NEW product type unrelated to the image
   Examples: "find me white pants", "show me red dresses", "I want sneakers"
   Key: Query mentions specific product type WITHOUT referring to the image

2. VISUAL_SEARCH - Find visually similar items (same look/appearance)
   Use when: User wants items that LOOK similar to the image
   Keywords: "similar", "like this", "same look", "find this item"

3. VISUAL_FILTER - Visual search + attribute filters
   Use when: User wants similar items BUT with specific attributes
   Keywords: "but in red", "similar but for men", "like this with filter"

4. VLM_ANALYSIS - Only analyze/describe the image
   Use when: User ONLY wants description, NO product recommendations
   Keywords: "what is this", "describe", "tell me about", "analyze style"

5. VLM_SEARCH - Analyze style then search for matching products
   Use when: User wants items matching the STYLE/VIBE, not exact appearance
   Keywords: "matching shoes", "pair with", "same style", "complete outfit"

CONTEXT:
Previous conversation:
{context}

USER QUERY: "{query}"

CRITICAL RULES (in priority order):
1. **NEW PRODUCT REQUEST**: If query mentions a specific product type (pants, dress, shoes, bag, etc.) 
   WITHOUT referencing the image (no "this", "similar", "like", "matching") → TEXT_RAG
2. If query asks ONLY for description → VLM_ANALYSIS
3. If query wants style transfer/matching items → VLM_SEARCH
4. If query mentions filters with image reference → VISUAL_FILTER
5. If query wants direct visual similarity → VISUAL_SEARCH
6. Default to VISUAL_SEARCH for ambiguous cases

Respond in JSON format:
{{"flow": "flow_name", "reasoning": "brief explanation"}}"""

    def _parse_response(self, content: str) -> FlowType:
        """Parse LLM response to extract flow type"""
        try:
            # Try to parse JSON
            content = content.strip()
            if content.startswith("```"):
                # Remove code blocks
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            data = json.loads(content)
            flow_name = data.get("flow", "").lower()

            # Map to enum
            flow_mapping = {
                "text_rag": FlowType.TEXT_RAG,
                "visual_search": FlowType.VISUAL_SEARCH,
                "visual_filter": FlowType.VISUAL_FILTER,
                "vlm_analysis": FlowType.VLM_ANALYSIS,
                "vlm_search": FlowType.VLM_SEARCH,
            }

            if flow_name in flow_mapping:
                return flow_mapping[flow_name]

            # Fallback: check if flow name contains keywords
            if "filter" in flow_name:
                return FlowType.VISUAL_FILTER
            elif "analysis" in flow_name or "describe" in flow_name:
                return FlowType.VLM_ANALYSIS
            elif "vlm" in flow_name or "style" in flow_name:
                return FlowType.VLM_SEARCH
            else:
                return FlowType.VISUAL_SEARCH

        except json.JSONDecodeError:
            # If JSON parsing fails, use heuristics on raw text
            logger.warning(f"Failed to parse JSON, using heuristics: {content}")
            content_lower = content.lower()

            if "filter" in content_lower:
                return FlowType.VISUAL_FILTER
            elif "analysis" in content_lower or "describe" in content_lower:
                return FlowType.VLM_ANALYSIS
            elif "vlm" in content_lower or "style" in content_lower:
                return FlowType.VLM_SEARCH
            else:
                return FlowType.VISUAL_SEARCH
