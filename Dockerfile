# Dockerfile
FROM python:3.12-slim

# System metadata
LABEL maintainer="Retail-AI MLOps Team"
LABEL version="0.7.0"
LABEL description="Retail-AI E-commerce Intelligence Engine"

# Set environment variables for Python and Poetry
# HF_HOME forces Hugging Face to cache models in a specific predictable folder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.2 \
    POETRY_VIRTUALENVS_CREATE=false \
    HF_HOME="/app/cache/huggingface"

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV/PIL or heavy ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry globally
RUN pip install "poetry==$POETRY_VERSION"

# Variable that defines the group: 'cpu' or 'gpu'
ARG INSTALL_GROUP=cpu

# Copy ONLY dependency manifests first to leverage Docker Layer Caching
COPY pyproject.toml poetry.lock ./

# Install production dependencies only (ignores pytest, mypy, locust, etc.)
RUN poetry install --only main --with ${INSTALL_GROUP} --no-interaction --no-ansi \
    && rm -rf ~/.cache/pypoetry \
    && rm -rf ~/.cache/pip

# Copy application source code and configurations
COPY src/ src/
COPY frontend/ frontend/

# Create the directories for volumes to ensure proper permissions
RUN mkdir -p /app/data/catalog /app/cache/huggingface

# Expose the API port
EXPOSE 8000

# Start the high-performance Uvicorn server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]