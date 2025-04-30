FROM python:3.12-slim

USER root

WORKDIR /app

COPY requirements.txt .

# Install system dependencies: ffmpeg and git
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git && \
    rm -rf /var/lib/apt/lists/*
    
RUN pip install  --upgrade --no-cache-dir -r requirements.txt




COPY ./scripts .
