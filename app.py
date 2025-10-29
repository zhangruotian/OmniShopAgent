"""
OmniShopAgent - Streamlit UI
Multi-modal fashion shopping assistant with conversational AI
"""

import logging
import uuid
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

from app.agents.orchestrator import AgentOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="OmniShopAgent - Fashion Shopping Assistant",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background-color: #e8f4f8;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f0f0f0;
        margin-right: 20%;
    }
    .message-avatar {
        font-size: 1.5rem;
        margin-right: 0.75rem;
    }
    .message-content {
        flex: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Initialize session state
def initialize_session():
    """Initialize session state variables"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = AgentOrchestrator(
            session_id=st.session_state.session_id
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None


def save_uploaded_image(uploaded_file) -> Optional[str]:
    """Save uploaded image to temp directory

    Args:
        uploaded_file: Streamlit uploaded file

    Returns:
        Path to saved image or None
    """
    if uploaded_file is None:
        return None

    try:
        # Create temp directory
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)

        # Save image
        image_path = temp_dir / f"{st.session_state.session_id}_{uploaded_file.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        logger.info(f"Saved uploaded image to {image_path}")
        return str(image_path)

    except Exception as e:
        logger.error(f"Error saving uploaded image: {e}")
        st.error(f"Failed to save image: {str(e)}")
        return None


def display_chat_message(role: str, content: str, image_path: Optional[str] = None):
    """Display a chat message

    Args:
        role: 'user' or 'assistant'
        content: Message content
        image_path: Optional path to image (for user messages)
    """
    avatar = "👤" if role == "user" else "🤖"
    message_class = f"{role}-message"

    with st.container():
        st.markdown(
            f'<div class="chat-message {message_class}">',
            unsafe_allow_html=True,
        )

        cols = st.columns([0.1, 0.9])

        with cols[0]:
            st.markdown(
                f'<div class="message-avatar">{avatar}</div>', unsafe_allow_html=True
            )

        with cols[1]:
            if image_path and role == "user":
                try:
                    image = Image.open(image_path)
                    st.image(image, width=200, caption="Uploaded Image")
                except Exception as e:
                    logger.error(f"Error displaying image: {e}")

            st.markdown(content)

        st.markdown("</div>", unsafe_allow_html=True)


def main():
    """Main Streamlit app"""
    initialize_session()

    # Header
    st.markdown(
        '<h1 class="main-header">👗 OmniShopAgent</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Your Intelligent Fashion Shopping Assistant - Powered by Multi-modal AI</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Settings")

        # Session info
        st.info(f"**Session ID:** `{st.session_state.session_id[:8]}...`")

        # Clear conversation
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.orchestrator.clear_history()
            st.session_state.messages = []
            st.session_state.uploaded_image = None
            st.success("Conversation cleared!")
            st.rerun()

        st.markdown("---")

        # Features
        st.markdown("### ✨ Features")
        st.markdown(
            """
        - 💬 **Text Search**: Describe what you want
        - 🖼️ **Image Search**: Upload product images
        - 🔍 **Visual Analysis**: AI analyzes image style
        - 🎯 **Smart Filtering**: Filter by color, gender, etc.
        - 💭 **Conversational**: Remembers context
        """
        )

        st.markdown("---")

        # Examples
        st.markdown("### 💡 Example Queries")
        st.markdown(
            """
        **Text Queries:**
        - "Show me blue casual shirts for men"
        - "Find red summer dresses"
        - "I need formal shoes for a wedding"

        **With Images:**
        - Upload image + "Find similar items"
        - Upload image + "Find this style but in red"
        - Upload image + "Analyze this product style"
        """
        )

        st.markdown("---")

        # About
        with st.expander("ℹ️ About"):
            st.markdown(
                """
            **OmniShopAgent** combines:
            - 🤖 GPT-4o-mini for reasoning
            - 🔤 OpenAI text embeddings
            - 👁️ CLIP vision model
            - 📊 Milvus vector database

            Built with LangChain, FastAPI, and Streamlit.
            """
            )

    # Main chat interface
    st.markdown("### 💬 Chat with Your Fashion Assistant")

    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(
            role=message["role"],
            content=message["content"],
            image_path=message.get("image_path"),
        )

    # Image upload section
    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 📸 Optional: Upload Product Image")

    with col2:
        if st.session_state.uploaded_image:
            if st.button("❌ Remove Image", use_container_width=True):
                st.session_state.uploaded_image = None
                st.rerun()

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a fashion product image for visual search or analysis",
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.session_state.uploaded_image = uploaded_file

    # Show uploaded image preview
    if st.session_state.uploaded_image:
        try:
            image = Image.open(st.session_state.uploaded_image)
            st.image(image, caption="Current Image", width=300)
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
            st.session_state.uploaded_image = None

    # Chat input
    user_query = st.chat_input(
        "Ask me anything about fashion products...",
        key="chat_input",
    )

    # Process user input
    if user_query:
        # Save uploaded image if present
        image_path = None
        if st.session_state.uploaded_image:
            image_path = save_uploaded_image(st.session_state.uploaded_image)

        # Add user message to history
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_query,
                "image_path": image_path,
            }
        )

        # Display user message
        display_chat_message("user", user_query, image_path)

        # Process with orchestrator
        with st.spinner("🤔 Thinking..."):
            try:
                result = st.session_state.orchestrator.process_query(
                    query=user_query,
                    image_path=image_path,
                )

                response = result["response"]
                intent = result.get("intent", "unknown")
                flow = result.get("flow", "unknown")

                # Add assistant message to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )

                # Display assistant message
                display_chat_message("assistant", response)

                # Show metadata in expander
                with st.expander("🔍 Debug Info"):
                    st.json(
                        {
                            "intent": intent,
                            "flow": flow,
                            "has_image": bool(image_path),
                        }
                    )

                # Clear uploaded image after processing
                st.session_state.uploaded_image = None

            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                error_msg = f"I apologize, I encountered an error: {str(e)}"

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                    }
                )

                display_chat_message("assistant", error_msg)

        # Rerun to update UI
        st.rerun()

    # Welcome message
    if not st.session_state.messages:
        st.markdown(
            """
        <div style="text-align: center; padding: 3rem; color: #666;">
            <h2>👋 Welcome to OmniShopAgent!</h2>
            <p>I'm your AI fashion shopping assistant. I can help you:</p>
            <ul style="list-style: none; padding: 0;">
                <li>🔎 Search for products by text description</li>
                <li>📸 Find visually similar items from images</li>
                <li>🎨 Analyze product styles and attributes</li>
                <li>🎯 Filter products by specific criteria</li>
            </ul>
            <p style="margin-top: 2rem;"><strong>Start by typing a message or uploading an image below!</strong></p>
        </div>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
