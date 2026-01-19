#!/bin/bash
# Complete Demo Runner Script
# Launches comprehensive demo with dashboard

echo "=========================================="
echo "  BondTrader Complete Demo"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

# Check if Streamlit is available
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Warning: Streamlit not installed. Dashboard will not launch."
    LAUNCH_DASHBOARD=""
else
    LAUNCH_DASHBOARD="--launch-dashboard"
fi

# Run comprehensive demo
echo "Running comprehensive demo..."
python scripts/comprehensive_demo.py $LAUNCH_DASHBOARD

echo ""
echo "=========================================="
echo "  Demo Complete!"
echo "=========================================="
echo ""
echo "To launch dashboard manually:"
echo "  streamlit run scripts/dashboard.py"
echo ""
echo "To view demo report:"
echo "  cat demo_report_*.md"
echo ""
