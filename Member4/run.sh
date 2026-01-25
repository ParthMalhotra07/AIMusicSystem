#!/bin/bash
# Launch script for AI Music Recommendation System (Member4)

set -e  # Exit on error

echo "🚀 Starting AI Music Recommendation System (Member4)..."
echo "=================================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if we're in the right directory
if [ ! -f "$SCRIPT_DIR/app/streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found in app/ directory"
    echo "   Make sure you're running this from the Member4 directory"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 not found. Please install Python 3.9+"
    exit 1
fi

echo "✅ Python: $(python3 --version)"
echo ""

# Check requirements
echo "📦 Checking dependencies..."
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "⚠️  Streamlit not found. Installing requirements..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    echo "✅ Dependencies OK"
fi

echo ""
echo "=================================================="
echo "🎵 LAUNCHING STREAMLIT DASHBOARD"
echo "=================================================="
echo ""
echo "🌐 Opening browser at: http://localhost:8501"
echo "📊 Dashboard: AI Music Recommendation System"
echo "🎨 Theme: Cyberpunk (Neon)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Launch Streamlit
cd "$SCRIPT_DIR"
streamlit run app/streamlit_app.py

