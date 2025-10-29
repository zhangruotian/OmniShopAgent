"""
OmniShopAgent - Streamlit UI
Multi-modal fashion shopping assistant with conversational AI
"""

import logging
import re
import uuid
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image, ImageOps

from app.agents.orchestrator import AgentOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="OmniShopAgent",
    page_icon="👗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS - ChatGPT-like style
st.markdown(
    """
    <style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Body and root container */
    .main .block-container {
        padding-bottom: 180px !important;
        padding-top: 2rem;
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Fixed input container at bottom */
    .fixed-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        border-top: 1px solid #e5e5e5;
        padding: 1rem 0;
        z-index: 1000;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
    }
    
    .fixed-input-container .block-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 0 1rem !important;
    }
    
    /* Message bubbles */
    .message {
        margin: 1rem 0;
        padding: 1rem 1.5rem;
        border-radius: 1rem;
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: transparent;
        margin: 0 0 1rem 0;
        padding: 0;
        border-radius: 0;
    }
    
    .assistant-message {
        background: white;
        border: 1px solid #e5e5e5;
        margin-right: 3rem;
    }
    
    /* Product cards - simplified */
    .stImage {
        border-radius: 0px;
        overflow: hidden;
    }
    
    .stImage img {
        transition: transform 0.2s;
    }
    
    .stImage:hover img {
        transform: scale(1.05);
    }
    
    /* Scroll to bottom behavior */
    html {
        scroll-behavior: smooth;
    }
    
    /* Header */
    .app-header {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 1rem;
        margin-bottom: 2rem;
    }
    
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .app-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Welcome screen */
    .welcome-container {
        text-align: center;
        padding: 4rem 2rem;
        color: #666;
    }
    
    .welcome-title {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 1rem;
    }
    
    .welcome-features {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: #f7f7f8;
        padding: 1.5rem;
        border-radius: 12px;
        transition: all 0.2s;
    }
    
    .feature-card:hover {
        background: #efefef;
        transform: translateY(-2px);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    
    /* Image preview */
    .image-preview {
        position: relative;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    .image-preview img {
        max-width: 200px;
        border-radius: 8px;
        border: 2px solid #e5e5e5;
    }
    
    .remove-image-btn {
        position: absolute;
        top: 5px;
        right: 5px;
        background: rgba(0,0,0,0.6);
        color: white;
        border: none;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        cursor: pointer;
        font-size: 14px;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #e5e5e5;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background: #f0f0f0;
        border-color: #d0d0d0;
    }
    
    /* Hide upload button label */
    .uploadedFile {
        display: none;
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

    if "show_image_upload" not in st.session_state:
        st.session_state.show_image_upload = False


def save_uploaded_image(uploaded_file) -> Optional[str]:
    """Save uploaded image to temp directory"""
    if uploaded_file is None:
        return None

    try:
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)

        image_path = temp_dir / f"{st.session_state.session_id}_{uploaded_file.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        logger.info(f"Saved uploaded image to {image_path}")
        return str(image_path)

    except Exception as e:
        logger.error(f"Error saving uploaded image: {e}")
        st.error(f"Failed to save image: {str(e)}")
        return None


def extract_products_from_response(response: str) -> list:
    """Extract product information from agent response

    Returns list of dicts with product info
    """
    products = []

    # Pattern to match product blocks in the response
    # Looking for ID, name, and other details
    lines = response.split("\n")
    current_product = {}

    for line in lines:
        line = line.strip()

        # Match product number (e.g., "1. Product Name" or "**1. Product Name**")
        if re.match(r"^\*?\*?\d+\.\s+", line):
            if current_product:
                products.append(current_product)
            current_product = {}
            # Extract product name
            name = re.sub(r"^\*?\*?\d+\.\s+", "", line)
            name = name.replace("**", "").strip()
            current_product["name"] = name

        # Match ID
        elif "ID:" in line or "id:" in line:
            id_match = re.search(r"(?:ID|id):\s*(\d+)", line)
            if id_match:
                current_product["id"] = id_match.group(1)

        # Match Category
        elif "Category:" in line:
            cat_match = re.search(r"Category:\s*(.+?)(?:\n|$)", line)
            if cat_match:
                current_product["category"] = cat_match.group(1).strip()

        # Match Color
        elif "Color:" in line:
            color_match = re.search(r"Color:\s*(\w+)", line)
            if color_match:
                current_product["color"] = color_match.group(1)

        # Match Gender
        elif "Gender:" in line:
            gender_match = re.search(r"Gender:\s*(\w+)", line)
            if gender_match:
                current_product["gender"] = gender_match.group(1)

        # Match Season
        elif "Season:" in line:
            season_match = re.search(r"Season:\s*(\w+)", line)
            if season_match:
                current_product["season"] = season_match.group(1)

        # Match Usage
        elif "Usage:" in line:
            usage_match = re.search(r"Usage:\s*(\w+)", line)
            if usage_match:
                current_product["usage"] = usage_match.group(1)

        # Match Similarity/Relevance score
        elif "Similarity:" in line or "Relevance:" in line:
            score_match = re.search(r"(?:Similarity|Relevance):\s*([\d.]+)%", line)
            if score_match:
                current_product["score"] = score_match.group(1)

    # Add last product
    if current_product:
        products.append(current_product)

    return products


def display_product_card(product: dict):
    """Display a product card with image and name only"""
    product_id = product.get("id", "")
    name = product.get("name", "Unknown Product")

    # Try to load image from data/images directory
    image_path = Path(f"data/images/{product_id}.jpg")

    # Create a compact card layout
    with st.container():
        if image_path.exists():
            try:
                img = Image.open(image_path)
                target_size = (180, 180)
                try:
                    img_processed = ImageOps.fit(img, target_size, method=Image.LANCZOS)
                except Exception:
                    img_processed = img.resize(target_size, Image.LANCZOS)

                st.image(img_processed, width=180, caption=name)
            except Exception:
                logger.warning(f"Failed to load image {image_path}")
                st.markdown(f"**📷 {name}**")
                st.caption(f"ID: {product_id}")
        else:
            logger.warning(f"Image not found: {image_path}")
            st.markdown(f"**📷 {name}**")
            st.caption(f"ID: {product_id}")


def display_message(message: dict):
    """Display a chat message"""
    role = message["role"]
    content = message["content"]
    image_path = message.get("image_path")
    tools_used = message.get("tools_used", [])

    if role == "user":
        st.markdown('<div class="message user-message">', unsafe_allow_html=True)

        if image_path and Path(image_path).exists():
            try:
                img = Image.open(image_path)
                st.image(img, width=200)
            except Exception:
                logger.warning(f"Failed to load user uploaded image: {image_path}")

        st.markdown(content)
        st.markdown("</div>", unsafe_allow_html=True)

    else:  # assistant
        # Display tools used (debug info)
        if tools_used:
            tools_badges = " ".join(
                [
                    f'<span style="background: #e3f2fd; color: #1976d2; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; margin-right: 0.25rem;">🔧 {tool}</span>'
                    for tool in tools_used
                ]
            )
            st.markdown(
                f'<div style="margin-bottom: 0.5rem;">{tools_badges}</div>',
                unsafe_allow_html=True,
            )

        # Extract and display products if any
        products = extract_products_from_response(content)

        if products:

            def parse_score(product: dict) -> float:
                score = product.get("score")
                if score is None:
                    return 0.0
                try:
                    return float(score)
                except (TypeError, ValueError):
                    return 0.0

            products = sorted(products, key=parse_score, reverse=True)[:3]

            # Display the text response first (without product details)
            text_lines = []
            for line in content.split("\n"):
                # Skip product detail lines
                if not any(
                    keyword in line
                    for keyword in [
                        "ID:",
                        "Category:",
                        "Color:",
                        "Gender:",
                        "Season:",
                        "Usage:",
                        "Similarity:",
                        "Relevance:",
                    ]
                ):
                    if not re.match(r"^\*?\*?\d+\.\s+", line):
                        text_lines.append(line)

            intro_text = "\n".join(text_lines).strip()
            if intro_text:
                st.markdown(intro_text)

            # Display product cards in grid (no separator)
            st.markdown("<br>", unsafe_allow_html=True)

            # Create 3-column grid for products
            cols_per_row = 3
            for i in range(0, len(products), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, product in enumerate(products[i : i + cols_per_row]):
                    with cols[j]:
                        display_product_card(product)
        else:
            # No products found, display full content
            st.markdown(content)

        st.markdown("</div>", unsafe_allow_html=True)


def display_welcome():
    """Display welcome screen"""

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <div class="feature-title">Text Search</div>
                <div>Describe what you want</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">📸</div>
                <div class="feature-title">Image Search</div>
                <div>Upload product photos</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Visual Analysis</div>
                <div>AI analyzes style</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-icon">💭</div>
                <div class="feature-title">Conversational</div>
                <div>Remembers context</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br><br>", unsafe_allow_html=True)


def main():
    """Main Streamlit app"""
    initialize_session()

    # Header
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">👗 OmniShopAgent</div>
            <div class="app-subtitle">AI Fashion Shopping Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar (collapsed by default, but accessible)
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            if "orchestrator" in st.session_state:
                st.session_state.orchestrator.shopping_agent.chat_history.clear()
            st.session_state.messages = []
            st.session_state.uploaded_image = None
            st.rerun()

        st.markdown("---")
        st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

    # Chat messages container
    messages_container = st.container()

    with messages_container:
        if not st.session_state.messages:
            display_welcome()
        else:
            for message in st.session_state.messages:
                display_message(message)

    # Fixed input area at bottom (using container to simulate fixed position)
    st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)

    input_container = st.container()

    with input_container:
        # Image upload area (shown when + is clicked)
        if st.session_state.show_image_upload:
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png"],
                key="file_uploader",
            )

            if uploaded_file:
                st.session_state.uploaded_image = uploaded_file
                # Show preview
                col1, col2 = st.columns([1, 4])
                with col1:
                    img = Image.open(uploaded_file)
                    st.image(img, width=100)
                with col2:
                    if st.button("❌ Remove"):
                        st.session_state.uploaded_image = None
                        st.session_state.show_image_upload = False
                        st.rerun()

        # Input row
        col1, col2 = st.columns([1, 12])

        with col1:
            # Image upload toggle button
            if st.button("➕", help="Add image", use_container_width=True):
                st.session_state.show_image_upload = (
                    not st.session_state.show_image_upload
                )
                st.rerun()

        with col2:
            # Text input
            user_query = st.chat_input(
                "Ask about fashion products...",
                key="chat_input",
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Process user input
    if user_query:
        # Ensure orchestrator is initialized
        if "orchestrator" not in st.session_state:
            st.error("Session not initialized. Please refresh the page.")
            st.stop()

        # Save uploaded image if present, or get from recent history
        image_path = None
        if st.session_state.uploaded_image:
            # User explicitly uploaded an image for this query
            image_path = save_uploaded_image(st.session_state.uploaded_image)
        else:
            # Check if query refers to a previous image
            query_lower = user_query.lower()
            if any(
                ref in query_lower
                for ref in [
                    "this",
                    "that",
                    "the image",
                    "the shirt",
                    "the product",
                    "it",
                ]
            ):
                # Find the most recent message with an image
                for msg in reversed(st.session_state.messages):
                    if msg.get("role") == "user" and msg.get("image_path"):
                        image_path = msg["image_path"]
                        logger.info(f"Using image from previous message: {image_path}")
                        break

        # Add user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_query,
                "image_path": image_path,
            }
        )

        # Display user message immediately
        with messages_container:
            display_message(st.session_state.messages[-1])

        # Process with orchestrator
        # Create a placeholder for thinking steps
        thinking_container = st.empty()

        try:
            # Show dynamic thinking steps
            agent = st.session_state.orchestrator.shopping_agent
            # Save orchestrator to local variable (can't access session_state in thread)
            orchestrator = st.session_state.orchestrator

            # Show initial thinking message
            with thinking_container.container():
                st.markdown("💭 **Thinking...**")

            # Process in background while monitoring steps
            import threading
            import time

            result = [None]
            error = [None]

            def process_query():
                try:
                    result[0] = orchestrator.process_query(
                        query=user_query,
                        image_path=image_path,
                    )
                except Exception as e:
                    error[0] = e

            # Start processing thread
            thread = threading.Thread(target=process_query)
            thread.start()

            # Monitor and display steps
            last_step = ""
            while thread.is_alive():
                current_step = getattr(agent, "current_step", "")
                if current_step and current_step != last_step:
                    with thinking_container.container():
                        st.markdown("💭 **Thinking...**")
                        st.markdown(f"_{current_step}_")
                    last_step = current_step
                time.sleep(0.1)

            thread.join()

            # Clear thinking container
            thinking_container.empty()

            # Check for errors
            if error[0]:
                raise error[0]

            if result[0] is None:
                raise Exception("No result returned from query processing")

            response = result[0]["response"]
            tools_used = result[0].get("tools_used", [])

            # Add assistant message
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response,
                    "tools_used": tools_used,
                }
            )

            # Clear uploaded image and hide upload area after sending
            st.session_state.uploaded_image = None
            st.session_state.show_image_upload = False

            # Auto-scroll to bottom with JavaScript
            st.markdown(
                """
                <script>
                window.scrollTo(0, document.body.scrollHeight);
                </script>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            error_msg = f"I apologize, I encountered an error: {str(e)}"

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg,
                }
            )

        # Rerun to update UI
        st.rerun()


if __name__ == "__main__":
    main()
