FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for tree-sitter and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create temp directory for repos
RUN mkdir -p /tmp/repos

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TEMP_DIR=/tmp/repos

# Run the worker
CMD ["python", "-m", "src.main"]
