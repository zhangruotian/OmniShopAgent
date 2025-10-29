"""
Intent Classification and Boundary Handling
Handles Flow 0: Classify user intents and handle edge cases
"""

import logging
from enum import Enum
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """User intent types"""

    SPECIFIC_SEARCH = "specific_search"  # Valid product search query
    TOO_VAGUE = "too_vague"  # Needs clarification
    OUT_OF_SCOPE = "out_of_scope"  # Not about fashion
    CHITCHAT = "chitchat"  # Greetings, thanks, casual talk


class IntentClassification(BaseModel):
    """Result of intent classification"""

    intent: IntentType
    confidence: float
    reasoning: str


class IntentClassifier:
    """Classifier for user query intents with boundary handling"""

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize intent classifier

        Args:
            openai_api_key: OpenAI API key (uses settings if not provided)
        """
        self.client = OpenAI(api_key=openai_api_key or settings.openai_api_key)
        self.model = settings.openai_model

    def classify(
        self, query: str, has_image: bool = False, conversation_history: list = None
    ) -> IntentClassification:
        """Classify user query intent

        Args:
            query: User's text query
            has_image: Whether user uploaded an image
            conversation_history: List of previous messages for context

        Returns:
            IntentClassification with intent type and reasoning
        """
        try:
            logger.info(f"Classifying intent for query: '{query}' (has_image={has_image})")

            prompt = self._build_classification_prompt(query, has_image, conversation_history)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )

            result = response.choices[0].message.content.strip()
            intent_type, reasoning = self._parse_classification_result(result)

            classification = IntentClassification(
                intent=intent_type,
                confidence=0.85,  # Can be refined with better parsing
                reasoning=reasoning,
            )

            logger.info(f"Intent classified as: {intent_type.value}")
            return classification

        except Exception as e:
            logger.error(f"Error classifying intent: {e}", exc_info=True)
            # Default to specific_search on error to allow processing
            return IntentClassification(
                intent=IntentType.SPECIFIC_SEARCH,
                confidence=0.5,
                reasoning="Error during classification, defaulting to search",
            )

    def _build_classification_prompt(self, query: str, has_image: bool, conversation_history: list = None) -> str:
        """Build prompt for intent classification"""
        image_context = (
            "\n- User has uploaded an image with their query"
            if has_image
            else "\n- User has NOT uploaded any image"
        )

        # Build conversation context
        history_context = ""
        if conversation_history and len(conversation_history) > 0:
            history_context = "\n\nPrevious conversation (last 3 exchanges):\n"
            # Get last 6 messages (3 exchanges)
            recent_messages = conversation_history[-6:]
            for msg in recent_messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:150]  # Truncate long messages
                history_context += f"{role.upper()}: {content}\n"
            history_context += "\nIMPORTANT: Consider this conversation history when classifying. If the current query refers to something from previous context (like 'this', 'that', 'matching', 'similar style'), it should be classified as SPECIFIC_SEARCH, not TOO_VAGUE."

        return f"""You are an intent classifier for a fashion e-commerce assistant.
Classify the user's query into ONE of these categories:

1. SPECIFIC_SEARCH: User wants to search for fashion products with clear intent
   Examples: "red dress", "running shoes for men", "find similar to this image"
   ALSO INCLUDES: Follow-up queries that reference previous conversation
   Examples: "what about in blue?", "find matching shoes", "similar but for women"
   
2. TOO_VAGUE: Query is too vague and needs clarification (ONLY if no conversation context)
   Examples: "recommend something", "I want to buy something", "show me products"
   
3. OUT_OF_SCOPE: Query is not about fashion products
   Examples: "I want to buy a phone", "recommend furniture", "best restaurants"
   
4. CHITCHAT: Greetings, thanks, casual conversation
   Examples: "hello", "thank you", "how are you", "bye"

Context:{image_context}{history_context}

Current User Query: "{query}"

Respond ONLY with the category name and brief reasoning in this format:
CATEGORY: <category_name>
REASONING: <brief explanation>
"""

    def _parse_classification_result(self, result: str) -> tuple[IntentType, str]:
        """Parse LLM classification result

        Args:
            result: LLM response text

        Returns:
            Tuple of (IntentType, reasoning)
        """
        lines = result.strip().split("\n")
        category = IntentType.SPECIFIC_SEARCH
        reasoning = "Unable to parse classification"

        for line in lines:
            if line.startswith("CATEGORY:"):
                category_str = line.replace("CATEGORY:", "").strip().lower()
                if "specific" in category_str or "search" in category_str:
                    category = IntentType.SPECIFIC_SEARCH
                elif "vague" in category_str:
                    category = IntentType.TOO_VAGUE
                elif "scope" in category_str:
                    category = IntentType.OUT_OF_SCOPE
                elif "chitchat" in category_str or "chat" in category_str:
                    category = IntentType.CHITCHAT
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        return category, reasoning


class BoundaryHandler:
    """Handles boundary cases and generates appropriate responses"""

    def handle(
        self,
        classification: IntentClassification,
        query: str,
        has_image: bool = False,
    ) -> str:
        """Generate response for boundary cases

        Args:
            classification: Intent classification result
            query: Original user query
            has_image: Whether user uploaded an image

        Returns:
            Appropriate response message
        """
        if classification.intent == IntentType.OUT_OF_SCOPE:
            return self._handle_out_of_scope(query)
        elif classification.intent == IntentType.TOO_VAGUE:
            return self._handle_too_vague(query, has_image)
        elif classification.intent == IntentType.CHITCHAT:
            return self._handle_chitchat(query)
        else:
            return ""  # No boundary handling needed

    def _handle_out_of_scope(self, query: str) -> str:
        """Handle out-of-scope queries"""
        return """I specialize in fashion products like clothing, shoes, bags, and accessories. 
I can help you find items like dresses, shirts, jeans, sneakers, watches, and more!

Could you tell me what fashion item you're looking for today?"""

    def _handle_too_vague(self, query: str, has_image: bool) -> str:
        """Handle too vague queries"""
        if has_image:
            return """I can see you've uploaded an image! Could you be more specific about what you're looking for?

For example:
- "Find similar items to this"
- "Find this style but in blue"
- "Find matching shoes for this outfit"
"""
        else:
            return """I'd love to help! Could you be more specific about what you're looking for?

For example:
- "Show me blue casual dresses"
- "I need formal shoes for a wedding"
- "Find red summer tops for women"
- Upload an image and ask "find similar items"
"""

    def _handle_chitchat(self, query: str) -> str:
        """Handle chitchat queries"""
        query_lower = query.lower()

        if any(greet in query_lower for greet in ["hello", "hi", "hey"]):
            return """Hello! 👋 I'm your fashion shopping assistant. 

I can help you:
- Search for products by description
- Find items similar to images you upload
- Filter products by color, style, gender, etc.

What are you looking for today?"""

        elif any(thanks in query_lower for thanks in ["thank", "thanks"]):
            return "You're welcome! Let me know if you need help finding anything else! 😊"

        elif any(bye in query_lower for bye in ["bye", "goodbye", "see you"]):
            return "Goodbye! Happy shopping! Come back anytime you need fashion advice! 👋"

        else:
            return """I'm here to help you find fashion products! 

Try asking me things like:
- "Show me summer dresses"
- "Find blue running shoes"
- Upload an image and say "find similar items"

What can I help you find?"""


# Singleton instances
_intent_classifier: Optional[IntentClassifier] = None
_boundary_handler: Optional[BoundaryHandler] = None


def get_intent_classifier() -> IntentClassifier:
    """Get singleton intent classifier instance"""
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier


def get_boundary_handler() -> BoundaryHandler:
    """Get singleton boundary handler instance"""
    global _boundary_handler
    if _boundary_handler is None:
        _boundary_handler = BoundaryHandler()
    return _boundary_handler

