# Running OmniShopAgent

Quick guide to run the OmniShopAgent Streamlit UI.

## Prerequisites

1. **Start CLIP Server** (in a separate terminal):
```bash
cd ~/Documents/clip
python -m clip_server torch-flow.yml
```

2. **Start Milvus** (if using Docker):
```bash
docker-compose up -d
```

3. **Set Environment Variables**:
```bash
# Make sure .env file has:
OPENAI_API_KEY=your_api_key_here
MILVUS_URI=http://localhost:19530  # For Milvus Standalone
```

## Run the Streamlit UI

**Method 1 (Recommended): Use the startup script**
```bash
cd /Users/zhangruotian/Documents/OmniShopAgent
./run_app.sh
```

**Method 2: Manual activation**
```bash
cd /Users/zhangruotian/Documents/OmniShopAgent
source venv/bin/activate
python -m streamlit run app.py
```

**Method 3: Direct path (if venv not activated)**
```bash
/Users/zhangruotian/Documents/OmniShopAgent/venv/bin/streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

> **Note**: Always use `python -m streamlit` or the startup script to ensure the correct virtual environment is used.

## Features

- **Text Search**: Type queries like "blue casual shirts for men"
- **Image Search**: Upload product images and ask "find similar"
- **Visual Analysis**: Upload images and ask "analyze this product style"
- **Filtering**: Ask "red dresses for women in summer"
- **Conversational**: Ask follow-up questions, agent remembers context

## Example Queries

### Text-Only
- "Show me blue casual shirts for men"
- "Find red summer dresses"
- "I need formal shoes for a wedding"

### With Images
- Upload image + "Find similar items"
- Upload image + "Find this style but in red"  
- Upload image + "Analyze this product and find similar cocktail dresses"

## Troubleshooting

**Port already in use**:
```bash
streamlit run app.py --server.port 8502
```

**CLIP server not running**:
```bash
# Make sure CLIP server is running on port 51000
ps aux | grep clip_server
```

**Milvus connection error**:
```bash
# Check Milvus is running
docker ps | grep milvus
```

## Stopping Services

```bash
# Stop Streamlit: Ctrl+C in terminal

# Stop CLIP server: Ctrl+C in CLIP terminal

# Stop Milvus:
docker-compose down
```

