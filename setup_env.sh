#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

VENV_DIR="venv"

echo "========================================"
echo "Setting up local development environment"
echo "========================================"

# 1. Create Virtual Environment
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists."
else
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# 2. Activate and Install Dependencies
echo "Activating virtual environment and installing dependencies..."
source "$VENV_DIR/bin/activate"

# Upgrade pip to ensure latest features
pip install --upgrade pip

# Install required packages from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
fi

# 3. Install the project in editable mode
echo "Installing project in editable mode (local package)..."
pip install -e .

echo "========================================"
echo "Setup complete!"
echo ""
echo "To start working, activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run the model:"
echo "  python two_towers.py"
echo "========================================"
