# OmniShopAgent

An autonomous multi-modal fashion shopping agent powered by **LangGraph** and **ReAct pattern**.

## Overview

OmniShopAgent autonomously decides which tools to call, maintains conversation state, and determines when to respond. Built with **LangGraph**, it uses agentic patterns for intelligent product discovery.

**Key Features:**
- Autonomous tool selection and execution
- Multi-modal search (text + image)
- Conversational context awareness
- Real-time visual analysis with GPT-4o-mini Vision

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Agent Framework** | LangGraph 0.2.74+ (StateGraph, MemorySaver) |
| **LLM** | GPT-4o-mini (reasoning, vision, tool calls) |
| **Text Embedding** | text-embedding-3-small |
| **Image Embedding** | CLIP ViT-B/32 (via clip-server) |
| **Vector Database** | Milvus |
| **Frontend** | Streamlit |
| **Dataset** | Kaggle Fashion Products (~44k items) |

## Architecture

**Agent Flow:**

```mermaid
graph LR
    START --> Agent
    Agent -->|Has tool_calls| Tools
    Agent -->|No tool_calls| END
    Tools --> Agent
    
    subgraph "Agent Node"
        A[Receive Messages] --> B[LLM Reasoning]
        B --> C{Need Tools?}
        C -->|Yes| D[Generate tool_calls]
        C -->|No| E[Generate Response]
    end
    
    subgraph "Tool Node"
        F[Execute Tools] --> G[Return ToolMessage]
    end
```

**Available Tools:**
- `search_products(query)` - Text-based semantic search
- `search_by_image(image_path)` - Visual similarity search  
- `analyze_image_style(image_path)` - VLM style analysis

## Examples

**Text Search:**
```
User: "winter coats for women"
Agent: search_products("winter coats women") → Returns 5 products
```

**Image Upload:**
```
User: [uploads sneaker photo] "find similar"
Agent: search_by_image(path) → Returns visually similar shoes
```

**Style Analysis + Search:**
```
User: [uploads vintage jacket] "what style is this? find matching pants"
Agent: analyze_image_style(path) → "Vintage denim bomber..."
       search_products("vintage pants casual") → Returns matching items
```

**Multi-turn Context:**
```
Turn 1: "show me red dresses"
Agent: search_products("red dresses") → Results

Turn 2: "make them formal"
Agent: [remembers context] → search_products("red formal dresses") → Results
```

**Complex Reasoning:**
```
User: [uploads office outfit] "I like the shirt but need something more casual"
Agent: analyze_image_style(path) → Extracts shirt details
       search_products("casual shirt [color] [style]") → Returns casual alternatives
```

## Installation

**Prerequisites:**
- Python 3.11+
- OpenAI API Key
- Docker & Docker Compose

### 1. Setup Environment
```bash
# Clone and install dependencies
git clone <repository-url>
cd OmniShopAgent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Download Dataset
Download the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) from Kaggle and extract to `./data/`:

```bash
# Using Kaggle CLI
kaggle datasets download -d paramaggarwal/fashion-product-images-dataset
unzip fashion-product-images-dataset.zip -d data/
```

Expected structure:
```
data/
├── images/       # ~44k product images
├── styles.csv    # Product metadata
└── images.csv    # Image filenames
```

### 3. Start Services

**Terminal 1 - Milvus (Vector Database):**
```bash
docker-compose up
```
Wait for: `Milvus Proxy started successfully`

**Terminal 2 - CLIP Server (Image Embeddings):**
```bash
python -m clip_server
```
Starts at `grpc://0.0.0.0:51000`

### 4. Index Data (One-time, ~10-15 min)
**Terminal 3:**
```bash
source venv/bin/activate
python scripts/index_data.py
```

This generates and stores text/image embeddings for all 44k products in Milvus.

### 5. Launch Application
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`
