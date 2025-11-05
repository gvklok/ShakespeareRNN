#!/bin/bash

# Quick Start Script for RNN Text Generator (PyTorch)
# This script sets up the environment and runs tests

echo "============================================================"
echo "RNN Text Generator - Quick Start Setup"
echo "============================================================"
echo ""

# Check if we're in the backend directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Please run this script from the backend directory"
    exit 1
fi

# Step 1: Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Step 1: Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Step 2: Activate virtual environment and install dependencies
echo ""
echo "Step 2: Installing dependencies..."
echo "This may take a few minutes..."
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Error installing dependencies"
    exit 1
fi

# Step 3: Run setup test
echo ""
echo "Step 3: Testing PyTorch installation..."
echo ""
python test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Setup Complete! 🎉"
    echo "============================================================"
    echo ""
    echo "Your environment is ready. Next steps:"
    echo ""
    echo "1. Make sure venv is activated:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Train the model:"
    echo "   python -m app.train"
    echo ""
    echo "3. Start the API server:"
    echo "   uvicorn app.main:app --reload"
    echo ""
    echo "For detailed instructions, see SETUP.md"
    echo "============================================================"
else
    echo ""
    echo "Setup test failed. Please check the errors above."
    exit 1
fi
