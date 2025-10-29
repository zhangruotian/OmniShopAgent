#!/bin/bash

# OmniShopAgent Startup Script
# This script ensures the correct virtual environment is used

cd "$(dirname "$0")"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please create it first: python -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if required services are running
echo "🔍 Checking services..."

# Check CLIP server
if ! nc -z localhost 51000 2>/dev/null; then
    echo "⚠️  WARNING: CLIP server not running on port 51000"
    echo "   Start it with: cd ~/Documents/clip && python -m clip_server torch-flow.yml"
fi

# Check Milvus
if ! nc -z localhost 19530 2>/dev/null; then
    echo "⚠️  WARNING: Milvus not running on port 19530"
    echo "   Start it with: docker-compose up -d"
fi

# Check .env file
if [ ! -f ".env" ]; then
    echo "⚠️  WARNING: .env file not found"
    echo "   Make sure to set OPENAI_API_KEY and other configs"
fi

echo ""
echo "✅ Starting OmniShopAgent..."
echo "🌐 App will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run Streamlit with the correct Python
exec python -m streamlit run app.py

