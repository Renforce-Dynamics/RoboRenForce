#!/bin/bash
# Quick activation script for RoboRenForce uv environment

# Activate uv virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✓ RoboRenForce environment activated (uv)"
    echo ""
    echo "Python: $(python --version)"
    echo "Location: $(which python)"
    echo ""
    echo "To deactivate: deactivate"
else
    echo "✗ Virtual environment not found!"
    echo "Run: uv sync"
    exit 1
fi
