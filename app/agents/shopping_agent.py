"""
Conversational Shopping Agent with LangGraph
True ReAct agent with autonomous tool calling and message accumulation
"""

import logging
from pathlib import Path
from typing import Optional, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import Annotated, TypedDict

from app.config import settings
from app.tools.search_tools import get_all_tools

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the shopping agent with message accumulation"""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_image_path: Optional[str]  # Track uploaded image


class ShoppingAgent:
    """True ReAct agent with autonomous decision making"""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or "default"

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            api_key=settings.openai_api_key,
        )

        # Get tools and bind to model
        self.tools = get_all_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Build graph
        self.graph = self._build_graph()

        logger.info(f"Shopping agent initialized for session: {self.session_id}")

    def _build_graph(self):
        """Build the LangGraph StateGraph"""

        # System prompt for the agent
        system_prompt = """You are an intelligent fashion shopping assistant. You can:
1. Search for products by text description (use search_products)
2. Find visually similar products from images (use search_by_image)
3. Analyze image style and attributes (use analyze_image_style)

When a user asks about products:
- For text queries: use search_products directly
- For image uploads: decide if you need to analyze_image_style first, then search
- You can call multiple tools in sequence if needed
- Always provide helpful, friendly responses

CRITICAL FORMATTING RULES:
When presenting product results, you MUST use this EXACT format for EACH product:

1. [Product Name]
   ID: [Product ID Number]
   Category: [Category]
   Color: [Color]
   Gender: [Gender]
   (Include Season, Usage, Relevance if available)

Example:
1. Puma Men White 3/4 Length Pants
   ID: 12345
   Category: Apparel > Bottomwear > Track Pants
   Color: White
   Gender: Men
   Season: Summer
   Usage: Sports
   Relevance: 95.2%

DO NOT skip the ID field! It is essential for displaying product images.
Be conversational in your introduction, but preserve the exact product format."""

        def agent_node(state: AgentState):
            """Agent decision node - decides which tools to call or when to respond"""
            messages = state["messages"]

            # Add system prompt if first message
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=system_prompt)] + list(messages)

            # Handle image context
            if state.get("current_image_path"):
                # Inject image path context for tool calls
                # The agent can reference this in its reasoning
                pass

            # Invoke LLM with tools
            response = self.llm_with_tools.invoke(messages)
            return {"messages": [response]}

        # Create tool node
        tool_node = ToolNode(self.tools)

        def should_continue(state: AgentState):
            """Determine if agent should continue or end"""
            messages = state["messages"]
            last_message = messages[-1]

            # If LLM made tool calls, continue to tools
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            # Otherwise, end (agent has final response)
            return END

        # Build graph
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue, ["tools", END])
        workflow.add_edge("tools", "agent")

        # Compile with memory
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)

    def chat(self, query: str, image_path: Optional[str] = None) -> dict:
        """Process user query with the agent

        Args:
            query: User's text query
            image_path: Optional path to uploaded image

        Returns:
            Dict with response and metadata
        """
        try:
            logger.info(
                f"[{self.session_id}] Processing: '{query}' (image={'Yes' if image_path else 'No'})"
            )

            # Validate image
            if image_path and not Path(image_path).exists():
                return {
                    "response": f"Error: Image file not found at '{image_path}'",
                    "error": True,
                }

            # Build input message
            message_content = query
            if image_path:
                message_content = f"{query}\n[User uploaded image: {image_path}]"

            # Invoke agent
            config = {"configurable": {"thread_id": self.session_id}}
            input_state = {
                "messages": [HumanMessage(content=message_content)],
                "current_image_path": image_path,
            }

            # Track tool calls
            tool_calls = []
            
            # Stream events to capture tool calls
            for event in self.graph.stream(input_state, config=config):
                logger.info(f"Event: {event}")
                
                # Check for agent node (tool calls)
                if "agent" in event:
                    agent_output = event["agent"]
                    if "messages" in agent_output:
                        for msg in agent_output["messages"]:
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    tool_calls.append({
                                        "name": tc["name"],
                                        "args": tc.get("args", {}),
                                    })
                
                # Check for tool node (tool results)
                if "tools" in event:
                    tools_output = event["tools"]
                    if "messages" in tools_output:
                        for i, msg in enumerate(tools_output["messages"]):
                            if i < len(tool_calls):
                                tool_calls[i]["result"] = str(msg.content)[:200] + "..."

            # Get final state
            final_state = self.graph.get_state(config)
            final_message = final_state.values["messages"][-1]
            response_text = final_message.content

            logger.info(f"[{self.session_id}] Response generated with {len(tool_calls)} tool calls")

            return {
                "response": response_text,
                "tool_calls": tool_calls,
                "error": False,
            }

        except Exception as e:
            logger.error(f"Error in agent chat: {e}", exc_info=True)
            return {
                "response": f"I apologize, I encountered an error: {str(e)}",
                "error": True,
            }

    def get_conversation_history(self) -> list:
        """Get conversation history for this session"""
        try:
            config = {"configurable": {"thread_id": self.session_id}}
            state = self.graph.get_state(config)

            if not state or not state.values.get("messages"):
                return []

            messages = state.values["messages"]
            result = []

            for msg in messages:
                # Skip system messages and tool messages
                if isinstance(msg, SystemMessage):
                    continue
                if hasattr(msg, "type") and msg.type in ["system", "tool"]:
                    continue

                role = "user" if msg.type == "human" else "assistant"
                result.append({"role": role, "content": msg.content})

            return result

        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []

    def clear_history(self):
        """Clear conversation history for this session"""
        # With MemorySaver, we can't easily clear, but we can log
        logger.info(f"[{self.session_id}] History clear requested")
        # In production, implement proper clearing or use new thread_id


def create_shopping_agent(session_id: Optional[str] = None) -> ShoppingAgent:
    """Factory function to create a shopping agent"""
    return ShoppingAgent(session_id=session_id)
