FROM python:3.12-slim

USER root

WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies: ffmpeg and git
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git && \
    rm -rf /var/lib/apt/lists/*


COPY ./scripts .