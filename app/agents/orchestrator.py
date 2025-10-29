"""
Main Orchestrator - Integrates Flow 0 (Intent) with Shopping Agent (Flows 1-4)
"""

import logging
from typing import Any, Dict, Optional

from app.agents.intent_classifier import (
    IntentType,
    get_boundary_handler,
    get_intent_classifier,
)
from app.agents.shopping_agent import ShoppingAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Main orchestrator that handles all flows (0-4)"""

    def __init__(self, session_id: Optional[str] = None):
        """Initialize orchestrator

        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id or "default"
        self.intent_classifier = get_intent_classifier()
        self.boundary_handler = get_boundary_handler()
        self.shopping_agent = ShoppingAgent(session_id=self.session_id)

        logger.info(f"Orchestrator initialized for session: {self.session_id}")

    def process_query(
        self,
        query: str,
        image_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process user query through all flows

        Flow 0: Intent classification and boundary handling
        Flow 1-4: Shopping agent with ReAct pattern

        Args:
            query: User's text query
            image_path: Optional path to uploaded image

        Returns:
            Dict with response and metadata
        """
        try:
            logger.info(f"[{self.session_id}] Processing query: '{query}'")

            # Get conversation history for context
            conversation_history = self.shopping_agent.get_conversation_history()

            # Flow 0: Intent Classification (with conversation context)
            classification = self.intent_classifier.classify(
                query=query,
                has_image=bool(image_path),
                conversation_history=conversation_history,
            )

            logger.info(
                f"[{self.session_id}] Intent: {classification.intent.value} "
                f"(confidence: {classification.confidence:.2f})"
            )

            # Handle boundary cases (but NOT if there's conversation history)
            if classification.intent != IntentType.SPECIFIC_SEARCH:
                # If there's conversation history, treat as specific search anyway
                # to allow the agent to use context
                if conversation_history and len(conversation_history) > 0:
                    logger.info(
                        f"[{self.session_id}] Has conversation history, treating as specific search"
                    )
                    classification.intent = IntentType.SPECIFIC_SEARCH
                else:
                    boundary_response = self.boundary_handler.handle(
                        classification=classification,
                        query=query,
                        has_image=bool(image_path),
                    )

                    return {
                        "response": boundary_response,
                        "intent": classification.intent.value,
                        "flow": "flow_0_boundary",
                        "error": False,
                    }

            # Flow 1-4: Shopping Agent (ReAct)
            result = self.shopping_agent.chat(query=query, image_path=image_path)

            return {
                "response": result["response"],
                "intent": classification.intent.value,
                "flow": self._infer_flow(query, image_path),
                "tools_used": self.shopping_agent.tools_used,  # Include tools used
                "error": result.get("error", False),
            }

        except Exception as e:
            logger.error(f"Error in orchestrator: {e}", exc_info=True)
            return {
                "response": f"I apologize, I encountered an error: {str(e)}",
                "intent": "error",
                "flow": "error",
                "error": True,
            }

    def _infer_flow(self, query: str, image_path: Optional[str]) -> str:
        """Infer which flow was likely used"""
        query_lower = query.lower()

        if image_path:
            if any(
                word in query_lower
                for word in ["analyze", "describe", "what is", "tell me about"]
            ):
                return "flow_4_react"  # VLM analysis
            elif any(
                word in query_lower for word in ["but", "in", "with", "color", "gender"]
            ):
                return "flow_3_visual_filter"  # Visual + filter
            else:
                return "flow_2_visual"  # Pure visual search

        # Text-only queries
        return "flow_1_text_rag"

    def get_conversation_history(self) -> list:
        """Get conversation history"""
        return self.shopping_agent.get_conversation_history()

    def clear_history(self):
        """Clear conversation history"""
        self.shopping_agent.clear_history()
        logger.info(f"[{self.session_id}] History cleared")
