FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Download and install `uv` binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /workspace