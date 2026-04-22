#!/bin/bash

echo "Setting up MLOps Environment..."

# Detect package manager and install dependencies
if command -v poetry &> /dev/null; then
    echo "Using Poetry..."
    poetry install
elif command -v uv &> /dev/null; then
    echo "Using UV..."
    uv venv
    source .venv/bin/activate
    uv pip install -e .
else
    echo "Using standard pip..."
    python -m venv .venv
    source .venv/bin/activate
    pip install fastapi uvicorn pydantic-settings pyyaml
fi

# Create .env from example if it doesn't exist
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Created .env from .env.example"
    else
        touch .env
        echo "Created empty .env"
    fi
fi

echo "Setup complete!"