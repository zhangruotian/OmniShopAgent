# OmniShopAgent

An intelligent multi-modal e-commerce search agent powered by LLMs, RAG, and visual understanding for fashion product discovery.

## Overview

OmniShopAgent combines Retrieval-Augmented Generation (RAG), multi-modal search, and conversational AI to create an intelligent shopping assistant. The system understands both text and image queries, maintains conversation context, and intelligently routes requests to appropriate processing pipelines.

**Key Features:**
- Multi-modal search (text and image queries)
- Conversational memory and context tracking
- Intelligent query routing with LangChain agents
- Intent classification and boundary handling
- ReAct pattern for complex multi-step reasoning
- Hybrid retrieval strategies (vector + metadata filtering)

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | GPT-4o-mini | Agent reasoning, VLM analysis, response generation |
| **Text Embedding** | text-embedding-3-small| Product description vectorization |
| **Image Embedding** | CLIP ViT-B/32 | Visual similarity search |
| **Vector Database** | Milvus Lite | Efficient similarity search for text & image vectors (embedded) |
| **Frontend** | Streamlit | Interactive web interface |
| **Agent Framework** | LangChain | Tool orchestration and flow routing |

## Dataset

Uses the **Fashion Product Images Dataset** from Kaggle (~44,000 products with attributes like color, category, gender, season, usage).

Download: `kaggle datasets download -d paramaggarwal/fashion-product-images-dataset`

## Architecture

```mermaid
graph TD
    %% --- 1. Core Components ---
    subgraph UserInput
        User["User<br/>(Identified by <b>Session ID</b>)"] 
    end
    
    subgraph ApplicationLayer
        API["<b>FastAPI Backend</b><br/>(Receives query + Session ID + optional image)"] 
    end
    
    subgraph OrchestrationLayer
        Agent["<b>Agent (LLM Router)</b><br/>Model: GPT-4o-mini<br/>Framework: LangChain<br/><i>'Which flow should I execute?'</i>"]
    end
    
    subgraph Tools
        T1["<b>Tool 1: VLM Reasoning</b><br/>GPT-4o-mini with Vision<br/><i>'Analyze image style/attributes'</i>"]
        T2["<b>Tool 2: Visual Search</b><br/>CLIP ViT-B/32 (Local)<br/><i>'Find visually similar products'</i>"]
        T3["<b>Tool 3: Text Search (RAG)</b><br/>text-embedding-3-small<br/><i>'Semantic text retrieval'</i>"]
    end

    subgraph DataStores[Data Stores]
        Milvus[("<b>Vector DB (Milvus)</b><br/>- Image Embeddings (512-dim)<br/>- Text Embeddings (1536-dim)<br/>~44k products")]
        SessionStore[("<b>Session Store (Streamlit)</b><br/>- Chat history per session<br/>- Conversation context<br/>- In-memory state")] 
    end

    %% --- 2. Start Flow ---
    User --> API
    API -- "Current Query + Session ID" --> IntentCheck["<b>Flow 0: Intent Classification</b><br/>- Out-of-scope?<br/>- Too vague?<br/>- Chitchat?<br/>- No match?"]
    
    IntentCheck -- "Boundary Case" --> BoundaryHandler["<b>Boundary Handler</b><br/>- Polite refusal<br/>- Ask clarification<br/>- Suggest alternatives<br/>- Casual response"]
    BoundaryHandler --> API
    
    IntentCheck -- "Valid Search Query" --> Agent
    SessionStore -- "Past Conversation<br/>(from Streamlit Session)" --> Agent 
    
    %% --- 3. Agent routes to different workflows ---

    %% --- Flow 1: Text RAG ---
    subgraph Flow1 ["Flow 1: Text RAG (e.g., 'a 100% cotton blue shirt')"]
        direction LR
        Agent -- "<b>Flow 1</b>" --> F1_Start(T3: Text Search)
        F1_Start -- "1. Query Vector" --> Milvus
        Milvus -- "2. Return Top K 'noisy' results<br/>(incl. 'Red Cotton Shirt')" --> F1_Augment
        F1_Augment["<b>Context Augmentation</b><br/>(Pass K results to LLM)"] --> F1_Generate
        F1_Generate["<b>Generation (Reasoning)</b><br/>LLM <b>reasons & filters</b> 'Red Shirt',<br/>then generates the answer"] --> API
    end

    %% --- Flow 2: Pure Visual Search ---
    subgraph Flow2 ["Flow 2: Pure Visual Search (e.g., [image] + 'find similar')"]
        direction LR
        Agent -- "<b>Flow 2</b>" --> F2_Start(T2: Visual Search)
        F2_Start -- "1. Image Vector" --> Milvus
        Milvus -- "2. Return Top K 'trusted' results" --> F2_Augment
        F2_Augment["<b>Context Augmentation</b><br/>(Pass K results to LLM)"] --> F2_Generate
        F2_Generate["<b>Generation (Presenting)</b><br/>LLM <b>presents</b> trusted results<br/>(no filtering needed)"] --> API
    end
    
    %% --- Flow 3: Visual Search + Metadata Filter ---
    subgraph Flow3 ["Flow 3: Visual Search + Filter (e.g., [image] + '...but in blue')"]
        direction LR
        Agent -- "<b>Flow 3</b>" --> F3_Start(T2: Visual Search)
        F3_Start -- "1. Image Vector" --> Milvus
        Milvus -- "2. Return Top 100 results" --> F3_Filter["<b>Metadata Filter</b><br/>(Filter in Python:<br/>WHERE color == 'blue')"]
        F3_Filter -- "3. Return filtered results" --> F3_Augment
        F3_Augment["<b>Context Augmentation</b><br/>(Pass 3 filtered results to LLM)"] --> F3_Generate
        F3_Generate["<b>Generation (Presenting)</b><br/>LLM formats the 3 results<br/>and generates the answer"] --> API
    end
    
    %% --- Flow 4: ReAct Loop (VLM -> RAG Chain) ---
    subgraph Flow4 ["Flow 4: ReAct Loop (e.g., [image] + 'find me cocktail dresses in this style')"]
        direction LR
        Agent -- "<b>Flow 4</b>" --> F4_Reason1["<b>Turn 1: Reason</b><br/>'I need the style first.<br/>I must call Tool 1 (VLM).'"]
        F4_Reason1 --> F4_Act1["<b>Turn 1: Act</b><br/>Call T1 (VLM)"]
        F4_Act1 --> T1
        T1 -- "<b>Observation:</b><br/>'Black A-line cocktail dress with lace'" --> F4_Reason2
        F4_Reason2["<b>Turn 2: Reason</b><br/>'OK, style is cocktail A-line.<br/>I must now call Tool 3 (RAG)<br/>to find similar cocktail dresses.'"]
        F4_Reason2 --> F4_Act2["<b>Turn 2: Act</b><br/>Call T3 (Text RAG)<br/>with new query"]
        F4_Act2 --> F4_RAGFlow("RAG Sub-Flow<br/>(Runs full Flow 1)")
        F4_RAGFlow -- "Final Answer" --> API
    end

    %% --- Flow 5: Conversational Memory ---
    subgraph Flow5 ["Flow 5: Conversational Memory (e.g., 'find matching shoes')"]
        direction LR
        Agent -- "<b>Flow 5</b>" --> F5_Reason["<b>Reason (with Memory)</b><br/>'Query is find matching shoes.<br/><b>Get history for Session ID.</b><br/>History says we just<br/>discussed 'casual white sneakers'.<br/>I must call Tool 3 (RAG).'"] 
        F5_Reason --> F5_Act["<b>Act</b><br/>Call T3 (Text RAG)<br/>with query:<br/>'casual white shoes for men'"]
        F5_Act --> F5_RAGFlow("RAG Sub-Flow<br/>(Runs full Flow 1)")
        F5_RAGFlow -- "Final Answer" --> API
    end
    
    %% --- 4. Final Response ---
    API -- "Final Answer" --> FinalResponse["User Response"]
    API -- "Save conversation<br/>(to Streamlit Session)" --> SessionStore 

    %% --- Style Definitions ---
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;
    class Agent fill:#e6f7ff,stroke:#0056b3,stroke-width:2px;
    class IntentCheck,BoundaryHandler fill:#fff0e6,stroke:#ff8c00,stroke-width:2px;
    class T1,T2,T3 fill:#fffbe6,stroke:#b8860b,stroke-width:2px;
    class Milvus,SessionStore fill:#e6ffe6,stroke:#006400,stroke-width:2px;
    class API,FinalResponse fill:#f0e6ff,stroke:#663399,stroke-width:2px;
    class F1_Generate,F4_RAGFlow,F5_RAGFlow fill:#ffebeb,stroke:b30000;
    class F2_Generate,F3_Generate fill:#ebfbee,stroke:006400;
    class F4_Reason1,F4_Act1,F4_Reason2,F4_Act2,F5_Reason,F5_Act fill:f5f5f5,stroke:444;
```

## Workflows

### Flow 0: Intent Classification & Boundary Handling
Classifies user intent before search execution to handle edge cases:
- **Out-of-Scope**: Non-fashion queries → Redirect to valid categories
- **Too Vague**: Unclear requests → Ask for clarification
- **Chitchat**: Greetings/casual talk → Engage and guide to shopping
- **Specific Search**: Valid queries → Route to Flows 1-4

LLM categorizes queries into: `specific_search`, `too_vague`, `out_of_scope`, or `chitchat`.

### Flow 1: Text-Based RAG Search
**Example**: *"Blue cotton shirt for casual wear"*

Text → Embedding → Milvus search → LLM filtering → Response

Text embeddings may be noisy; LLM filters false positives.

### Flow 2: Pure Visual Search
**Example**: [Image] + *"Find similar items"*

Image → CLIP embedding → Milvus search → Direct results

CLIP provides accurate visual similarity without filtering.

### Flow 3: Visual Search + Attribute Filtering
**Example**: [Image] + *"Similar but in red"*

Image → CLIP → Top-100 results → Metadata filter → Filtered results

Combines visual similarity with attribute constraints.

### Flow 4: ReAct Loop (VLM → RAG)
**Example**: [Image] + *"Find cocktail dresses in this style"*

Image → VLM analysis (style/attributes) → Text search with extracted attributes → Results

Multi-step reasoning: analyze style first, then search by description.

### Flow 5: Conversational Memory
**Example**: 
- Turn 1: *"Show me white sneakers"*
- Turn 2: *"Now find a matching backpack"* → Uses context from Turn 1

Retrieves chat history → Understands context → Augments query → Execute search

Enables natural multi-turn conversations via Streamlit session state.

## Installation

**Prerequisites:**
- Python 3.11+ or 3.13+
- OpenAI API Key
- Docker & Docker Compose (for Milvus)

**Quick Start:**

### 1. Clone and Install Dependencies
```bash
git clone <repository-url>
cd OmniShopAgent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
CLIP_SERVER_URL=grpc://localhost:51000
```

### 3. Download Dataset
```bash
# Option 1: Use Kaggle CLI (requires Kaggle account)
kaggle datasets download -d paramaggarwal/fashion-product-images-dataset
unzip fashion-product-images-dataset.zip -d data/

# Option 2: Use download script
python scripts/download_dataset.py
```

The dataset should be extracted to `./data/` with the following structure:
```
data/
├── images/       # Product images
├── styles.csv    # Product metadata
└── images.csv    # Image paths
```

### 4. Start Required Services

**Terminal 1 - Start Milvus Vector Database:**
```bash
docker-compose up
```

Wait until you see `Milvus Proxy started successfully` in the logs.

**Terminal 2 - Start CLIP Server:**
```bash
# Install CLIP server if not already installed
pip install clip-server

# Start CLIP server on port 51000
python -m clip_server
```

The CLIP server will start at `grpc://0.0.0.0:51000` by default.

### 5. Index Product Data (One-time Setup)
In a new terminal:
```bash
source venv/bin/activate  # Activate virtual environment

# Generate and store embeddings in Milvus
# This will take ~10-15 minutes for 44k products
python scripts/index_data.py
```

You should see progress like:
```
Processing batch 1/440...
Processing batch 2/440...
...
Indexing complete! Indexed 44,000 products.
```

### 6. Launch the Application
```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501` 










